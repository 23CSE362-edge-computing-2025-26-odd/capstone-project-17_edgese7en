#!/usr/bin/env python3
# Full script saved here; please open the RTF for instructions and the separate Python file for the runnable code.
# For convenience the full runnable Python script is written below.


#!/usr/bin/env python3
"""
simpy_video_traffic_sim_full.py
(Full runnable script — same content as carefully provided in the RTF)
"""
import argparse
import simpy
import cv2
import numpy as np
import random
import time
import csv
import os
from collections import defaultdict, deque
import pandas as pd
import matplotlib.pyplot as plt

# Optional rich-based live dashboard
try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    RICH_OK = True
    console = Console()
except Exception:
    RICH_OK = False

def extract_tasks_from_video(video_source, frame_skip=8, urgent_prob=0.1, payload_kb=50, debug=False, max_frames=None):
    tasks = []
    try:
        idx = int(video_source)
        cap = cv2.VideoCapture(idx)
    except Exception:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    task_id = 0
    max_frames_local = max_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames_local and frame_idx > max_frames_local:
            break
        if frame_idx % frame_skip != 0:
            continue
        timestamp = frame_idx / fps
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg = bg_sub.apply(gray)
        th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vehicles = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:
                vehicles += 1
        if debug:
            print(f"[frame {frame_idx}] time {timestamp:.2f}s vehicles~{vehicles}")
        for v in range(vehicles):
            is_urgent = (random.random() < urgent_prob)
            priority_str = "urgent" if is_urgent else "normal"
            priority_int = 0 if is_urgent else 1
            work_mi = 2000 if is_urgent else 500
            deadline_ms = 2000 if is_urgent else 8000
            task = {
                "task_id": f"t{task_id}",
                "arrival_time": timestamp,
                "priority_str": priority_str,
                "priority_int": priority_int,
                "work_mi": work_mi,
                "deadline_ms": deadline_ms,
                "payload_kb": payload_kb,
                "source_camera": "cam0"
            }
            tasks.append(task)
            task_id += 1

    cap.release()
    print(f"Extracted {len(tasks)} tasks from video (sampled every {frame_skip} frames). fps={fps}")
    return tasks, fps

class StatsCollector:
    def __init__(self):
        self.total_created = 0
        self.enqueued = 0
        self.processed = 0
        self.dropped = 0
        self.deadline_misses = 0
        self.latencies = []
        self.per_priority = defaultdict(list)
        self.offloaded_count = 0
        self.edge_processed_count = 0
        self.timeseries = []

    def record_processed(self, task, where, latency):
        self.processed += 1
        self.latencies.append((task['task_id'], where, task['priority_str'], latency))
        self.per_priority[task['priority_str']].append(latency)
        if where == "cloud":
            self.offloaded_count += 1
        else:
            self.edge_processed_count += 1

    def record_dropped(self, task):
        self.dropped += 1

    def record_deadline_miss(self):
        self.deadline_misses += 1

    def snapshot(self, env, nodes):
        queued = sum(len(n.queue.items) for n in nodes)
        proc = sum(n.currently_processing for n in nodes)
        self.timeseries.append((env.now, queued, proc, 0))

class SimpleNetwork:
    def __init__(self, env, bandwidth_mbps=5.0, max_parallel=3):
        self.env = env
        self.bandwidth_mbps = bandwidth_mbps
        self.max_parallel = max_parallel
        self.slot_resource = simpy.Resource(env, capacity=max_parallel)
        self.active_transfers = 0
        self.bandwidth_usage_timeseries = []

    def transfer(self, payload_kb):
        with self.slot_resource.request() as req:
            yield req
            self.active_transfers += 1
            effective_kbps = (self.bandwidth_mbps * 1000.0) / float(self.max_parallel)
            transfer_seconds = (payload_kb * 8.0) / effective_kbps
            self.bandwidth_usage_timeseries.append((self.env.now, self.active_transfers))
            yield self.env.timeout(transfer_seconds)
            self.active_transfers -= 1

class EdgeNode:
    def __init__(self, env, node_id, mips, max_queue, network, cloud_node, stats: StatsCollector, scheduler_policy):
        self.env = env
        self.node_id = node_id
        self.mips = mips
        self.queue = simpy.PriorityStore(env, capacity=max_queue)
        self.currently_processing = 0
        self.network = network
        self.cloud_node = cloud_node
        self.stats = stats
        self.seq = 0
        self.scheduler_policy = scheduler_policy
        self.proc_proc = env.process(self._processor_loop())

    def handle_arrival(self, task):
        self.stats.total_created += 1
        if len(self.queue.items) < self.queue.capacity:
            self.seq += 1
            task['enqueue_time'] = self.env.now
            self.queue.put((task['priority_int'], self.seq, task))
            self.stats.enqueued += 1
        else:
            do_offload = self.scheduler_policy(self, task)
            if do_offload:
                self.env.process(self._offload_to_cloud(task))
            else:
                self.stats.record_dropped(task)

    def _estimated_local_wait(self, task):
        total_wait_mi = 0
        for (p, s, t) in list(self.queue.items):
            total_wait_mi += t['work_mi']
        estimated_wait_sec = total_wait_mi / float(self.mips) if self.mips>0 else 9999.0
        own_proc_sec = task['work_mi'] / float(self.mips) if self.mips>0 else 9999.0
        return estimated_wait_sec + own_proc_sec

    def _estimate_cloud_time(self, task):
        transfer_time_est = (task['payload_kb'] * 8.0) / (self.network.bandwidth_mbps * 1000.0)
        proc_est = task['work_mi'] / float(self.cloud_node.mips) if self.cloud_node.mips>0 else 9999.0
        return transfer_time_est + proc_est + 0.05

    def _processor_loop(self):
        while True:
            pri, seq, task = yield self.queue.get()
            task_start = self.env.now
            self.currently_processing += 1
            proc_time = task['work_mi'] / float(self.mips)
            yield self.env.timeout(proc_time)
            finish_time = self.env.now
            latency = (finish_time - task['arrival_time'])
            deadline_ok = (latency * 1000.0) <= task.get('deadline_ms', 9999999)
            if not deadline_ok:
                self.stats.record_deadline_miss()
            self.stats.record_processed(task, "edge", latency)
            self.currently_processing -= 1

    def _offload_to_cloud(self, task):
        offload_start = self.env.now
        yield self.env.process(self.network.transfer(task['payload_kb']))
        yield self.env.process(self.cloud_node.process_task(task))
        offload_finish = self.env.now
        latency = offload_finish - task['arrival_time']
        deadline_ok = (latency * 1000.0) <= task.get('deadline_ms', 9999999)
        if not deadline_ok:
            self.stats.record_deadline_miss()
        self.stats.record_processed(task, "cloud", latency)

class CloudNode:
    def __init__(self, env, mips, concurrency=4):
        self.env = env
        self.mips = mips
        self.proc_slots = simpy.Resource(env, capacity=concurrency)

    def process_task(self, task):
        with self.proc_slots.request() as req:
            yield req
            proc_time = task['work_mi'] / float(self.mips)
            yield self.env.timeout(proc_time)

def default_scheduler_policy(edge_node: EdgeNode, task):
    remaining_deadline_s = task['deadline_ms'] / 1000.0
    est_local = edge_node._estimated_local_wait(task)
    est_cloud = edge_node._estimate_cloud_time(task)
    if est_local <= remaining_deadline_s:
        return False
    return True

def run_simulation_from_tasks(tasks, nodes_count=3, edge_mips=800, cloud_mips=5000,
                              bandwidth_mbps=5.0, max_parallel=3, max_queue=8,
                              sim_time=300, report_interval=2.0, output_prefix="result",
                              scheduler_policy=default_scheduler_policy):
    env = simpy.Environment()
    stats = StatsCollector()
    network = SimpleNetwork(env, bandwidth_mbps=bandwidth_mbps, max_parallel=max_parallel)
    cloud = CloudNode(env, mips=cloud_mips, concurrency=8)
    edge_nodes = []
    for i in range(nodes_count):
        n = EdgeNode(env, f"edge-{i}", mips=edge_mips, max_queue=max_queue,
                     network=network, cloud_node=cloud, stats=stats, scheduler_policy=scheduler_policy)
        edge_nodes.append(n)

    def task_arrival_gen(env, tasks):
        tasks_sorted = sorted(tasks, key=lambda x: x['arrival_time'])
        for t in tasks_sorted:
            now = env.now
            to_wait = t['arrival_time'] - now
            if to_wait > 0:
                yield env.timeout(to_wait)
            node_index = hash(t['source_camera']) % len(edge_nodes)
            t['arrival_time'] = env.now
            edge_nodes[node_index].handle_arrival(t)
        return

    env.process(task_arrival_gen(env, tasks))

    def reporter(env, stats, edge_nodes):
        while True:
            yield env.timeout(report_interval)
            stats.snapshot(env, edge_nodes)
            if RICH_OK:
                table = None
                from rich.table import Table
                table = Table(title=f"Sim t={env.now:.2f}s")
                table.add_column("Edge Node")
                table.add_column("Queue Len")
                table.add_column("Processing")
                for n in edge_nodes:
                    table.add_row(n.node_id, str(len(n.queue.items)), str(n.currently_processing))
                from rich.console import Console
                console = Console()
                console.clear()
                console.print(table)
            else:
                print(f"t={env.now:.2f}s | created={stats.total_created} enq={stats.enqueued} proc={stats.processed} drop={stats.dropped} off={stats.offloaded_count}")
                for n in edge_nodes:
                    print(f"  {n.node_id}: queue={len(n.queue.items)} proc={n.currently_processing}")

    env.process(reporter(env, stats, edge_nodes))
    start_wall = time.time()
    env.run(until=sim_time)
    wall_time = time.time() - start_wall

    results_dir = f"{output_prefix}_sim_results"
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"summary_nodes{nodes_count}_mips{edge_mips}_bw{bandwidth_mbps}.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nodes","edge_mips","cloud_mips","bandwidth_mbps","max_parallel","sim_time","created","enqueued","processed","dropped","offloaded","deadline_misses","wall_seconds"])
        writer.writerow([nodes_count, edge_mips, cloud_mips, bandwidth_mbps, max_parallel, sim_time,
                         stats.total_created, stats.enqueued, stats.processed, stats.dropped,
                         stats.offloaded_count, stats.deadline_misses, wall_time])
    events_path = os.path.join(results_dir, f"events_nodes{nodes_count}_mips{edge_mips}_bw{bandwidth_mbps}.csv")
    with open(events_path, "w", newline="") as f:
        keys = ["task_id","where","priority","latency"]
        writer = csv.writer(f)
        writer.writerow(keys)
        for (tid, where, pri, lat) in stats.latencies:
            writer.writerow([tid, where, pri, lat])
    print("Saved summary to", summary_path)
    print("Saved events to", events_path)
    try:
        df_ev = pd.read_csv(events_path)
        plt.figure(figsize=(7,4))
        df_ev['latency'].hist(bins=40)
        plt.title("Latency distribution (all processed tasks)")
        plt.xlabel("seconds")
        plt.ylabel("count")
        p1 = os.path.join(results_dir, "latency_hist.png")
        plt.tight_layout()
        plt.savefig(p1)
        plt.figure(figsize=(6,4))
        counts = df_ev['where'].value_counts()
        counts.plot(kind='bar')
        plt.title("Processed: Edge vs Cloud")
        plt.ylabel("count")
        p2 = os.path.join(results_dir, "processed_edge_cloud.png")
        plt.tight_layout()
        plt.savefig(p2)
        plt.figure(figsize=(6,4))
        df_ev.groupby('priority')['latency'].mean().plot(kind='bar')
        plt.title("Average latency by priority")
        plt.ylabel("seconds")
        p3 = os.path.join(results_dir, "avg_latency_by_priority.png")
        plt.tight_layout()
        plt.savefig(p3)
        print("Saved plots:", p1, p2, p3)
    except Exception as e:
        print("Plotting failed:", e)
    return stats

def main():
    parser = argparse.ArgumentParser(description="SimPy video-driven edge vs cloud traffic processing simulator")
    parser.add_argument("C:\\Users\\ADMIN\\Downloads\\rev2vido.mp4", required=True, help="video file path or webcam index (0)")
    parser.add_argument("--frame-skip", type=int, default=8, help="sample every N frames")
    parser.add_argument("--nodes", type=int, default=3, help="number of edge nodes")
    parser.add_argument("--edge-mips", type=float, default=800.0, help="edge node MIPS")
    parser.add_argument("--cloud-mips", type=float, default=5000.0, help="cloud MIPS")
    parser.add_argument("--bandwidth", type=float, default=5.0, help="network bandwidth Mbps")
    parser.add_argument("--max-parallel", type=int, default=3, help="parallel transfers allowed")
    parser.add_argument("--max-queue", type=int, default=8, help="max queue per edge")
    parser.add_argument("--sim-time", type=float, default=300.0, help="simulation duration in seconds")
    parser.add_argument("--payload-kb", type=float, default=50.0, help="payload per task in KB")
    parser.add_argument("--urgent-prob", type=float, default=0.08, help="probability a vehicle is urgent")
    parser.add_argument("--report-interval", type=float, default=3.0, help="reporting interval in sim seconds")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="limit frames to parse (for quick runs)")
    args = parser.parse_args()

    print("Extracting tasks from video (this may take a while for large videos)...")
    tasks, fps = extract_tasks_from_video(args.video, frame_skip=args.frame_skip, urgent_prob=args.urgent_prob,
                                         payload_kb=args.payload_kb, debug=args.debug, max_frames=args.max_frames)
    if len(tasks) == 0:
        print("No tasks detected. Exiting.")
        return
    last_arrival = max(t['arrival_time'] for t in tasks)
    sim_time = args.sim_time
    if sim_time <= 0:
        sim_time = last_arrival + 30.0

    stats = run_simulation_from_tasks(tasks, nodes_count=args.nodes, edge_mips=args.edge_mips,
                                     cloud_mips=args.cloud_mips,
                                     bandwidth_mbps=args.bandwidth, max_parallel=args.max_parallel,
                                     max_queue=args.max_queue, sim_time=sim_time, report_interval=args.report_interval,
                                     output_prefix="video_sim", scheduler_policy=default_scheduler_policy)

    print("=== FINAL SUMMARY ===")
    print("Total created:", stats.total_created)
    print("Enqueued:", stats.enqueued)
    print("Processed:", stats.processed)
    print("Dropped:", stats.dropped)
    print("Offloaded:", stats.offloaded_count)
    print("Edge processed:", stats.edge_processed_count)
    print("Deadline misses:", stats.deadline_misses)
    if stats.per_priority:
        for p in stats.per_priority:
            vals = stats.per_priority[p]
            print(f"Priority {p}: count={len(vals)} avg_latency={sum(vals)/len(vals):.3f}s")
    print("Results saved in directories matching 'video_sim_sim_results*'")

if __name__ == "__main__":
    main()
