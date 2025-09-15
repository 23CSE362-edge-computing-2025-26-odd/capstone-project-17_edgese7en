r"""
simpy_edge_cloud.py

Simple SimPy edge/cloud simulator for tasks produced from video_to_tasks.py

Usage example:
python .\simpy_edge_cloud.py --tasks .\tasks.csv --nodes 3 --edge-mips 800 --cloud-mips 5000 \
    --bandwidth_mbps 8 --max-queue 4 --sim-time 600 --offload-policy queue --out-log .\results.csv
"""

import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simpy
from tqdm import tqdm

# ---------------------------
# Robust task loader
# ---------------------------
def load_tasks(csv_path):
    """
    Load tasks CSV robustly.
    - Accepts flexible column names (arrival_time / time / timestamp, mi / work / cycles, data_size / size, node_id)
    - Converts arrival times to floats (relative to first arrival)
    - Parses MI and data_size defensively (handles commas, empty strings, non-numeric)
    """
    df = pd.read_csv(csv_path)
    # Normalize column names to lower for flexible lookup
    cols = {c.lower(): c for c in df.columns}

    # arrival time column
    if 'arrival_time' in cols:
        arrival_col = cols['arrival_time']
    elif 'time' in cols:
        arrival_col = cols['time']
    elif 'timestamp' in cols:
        arrival_col = cols['timestamp']
    else:
        arrival_col = df.columns[0]

    # MI column (work amount). Common names: mi, work, cycles
    if 'mi' in cols:
        mi_col = cols['mi']
    elif 'work' in cols:
        mi_col = cols['work']
    elif 'cycles' in cols:
        mi_col = cols['cycles']
    else:
        mi_col = df.columns[1] if len(df.columns) > 1 else None

    # data size column
    if 'data_size' in cols:
        data_col = cols['data_size']
    elif 'size' in cols:
        data_col = cols['size']
    else:
        data_col = None

    # node id
    node_id_col = cols.get('node_id', None)

    def safe_float(val, default=None):
        if val is None:
            return default
        if isinstance(val, (int, float, np.floating, np.integer)):
            return float(val)
        s = str(val).strip()
        if s == '' or s.lower() in ['nan', 'none', 'na', 'n/a', '-']:
            return default
        # remove commas like "1,000"
        s = s.replace(',', '')
        try:
            return float(s)
        except Exception:
            # maybe there are extra characters, try to extract leading numeric part
            num = ''
            dot_seen = False
            sign_allowed = True
            for ch in s:
                if ch.isdigit():
                    num += ch
                elif ch == '.' and not dot_seen:
                    num += ch
                    dot_seen = True
                elif (ch in '+-') and sign_allowed:
                    num += ch
                    sign_allowed = False
                else:
                    break
            try:
                return float(num) if num not in ['', '+', '-'] else default
            except Exception:
                return default

    tasks = []
    for i, row in df.iterrows():
        # arrival
        arrival = None
        try:
            raw_arr = row[arrival_col]
            # handle datetimes
            if pd.isna(raw_arr):
                arrival = float(i)
            else:
                try:
                    arrival = float(raw_arr)
                except Exception:
                    try:
                        arrival = float(pd.to_datetime(raw_arr).timestamp())
                    except Exception:
                        arrival = float(i)
        except Exception:
            arrival = float(i)

        # mi
        mi_val = None
        if mi_col is not None:
            try:
                mi_raw = row[mi_col]
            except Exception:
                mi_raw = None
            mi_val = safe_float(mi_raw, default=1000.0)
            if mi_val is None:
                print(f"[load_tasks] warning: row {i} invalid MI '{mi_raw}', using default 1000.0")
                mi_val = 1000.0
        else:
            mi_val = 1000.0

        # data size (MB)
        data_mb = None
        if data_col is not None:
            try:
                data_raw = row[data_col]
            except Exception:
                data_raw = None
            data_mb = safe_float(data_raw, default=0.05)
            if data_mb is None:
                data_mb = 0.05
        else:
            data_mb = 0.05

        # node id
        try:
            node_id = str(row[node_id_col]) if node_id_col is not None and not pd.isna(row[node_id_col]) else None
        except Exception:
            node_id = None

        tid = row.get('task_id', None) if 'task_id' in row.index else f"t{i}"

        tasks.append({
            'id': tid,
            'arrival': arrival,
            'mi': mi_val,
            'data_mb': data_mb,
            'node_id': node_id,
            'raw': row.to_dict()
        })

    # Sort & normalize arrivals
    tasks.sort(key=lambda x: x['arrival'])
    if len(tasks) > 0:
        t0 = tasks[0]['arrival']
        for t in tasks:
            t['arrival'] = max(0.0, float(t['arrival'] - t0))
    return tasks

# ---------------------------
# Node and Cloud worker definitions
# ---------------------------
class EdgeNode:
    def __init__(self, env, node_index, mips, max_queue):
        self.env = env
        self.node_index = node_index
        self.mips = float(mips)  # MI per second
        # Use Store for FIFO queue; capacity enforced via capacity attribute if provided
        if max_queue is None:
            self.queue = simpy.Store(env)  # infinite
        else:
            # simpy.Store accepts capacity (SimPy 4), but some versions use _capacity internally
            try:
                self.queue = simpy.Store(env, capacity=int(max_queue))
            except TypeError:
                # fallback: create Store and attach capacity attribute
                self.queue = simpy.Store(env)
                setattr(self.queue, 'capacity', int(max_queue))
        self.process = env.process(self.run())

    def __repr__(self):
        return f"EdgeNode({self.node_index}, mips={self.mips})"

    def queue_len(self):
        return len(getattr(self.queue, 'items', []))

    def queue_capacity(self):
        capacity = getattr(self.queue, 'capacity', None)
        if capacity is None:
            capacity = getattr(self.queue, '_capacity', None)
        if capacity is None:
            return float('inf')
        return capacity

    def enqueue(self, task):
        # Put without yielding; store.put returns an event but we don't wait here
        self.queue.put(task)

    def run(self):
        while True:
            task = yield self.queue.get()
            # record start processing time on edge
            task['started_at'] = self.env.now
            # compute processing time (seconds) = MI / MIPS
            proc_time = float(task['mi']) / max(1.0, self.mips)
            # simulate processing; for realism we add a tiny jitter
            yield self.env.timeout(proc_time)
            # mark finish
            task['finished_at'] = self.env.now
            task['processed_by'] = f"edge-{self.node_index}"
            # append to global results
            SIM_RESULTS.append(task)

class Cloud:
    def __init__(self, env, mips, network_bandwidth_mbps):
        self.env = env
        self.mips = float(mips)
        self.bandwidth_mbps = float(network_bandwidth_mbps)  # system-wide bandwidth for simplistic modeling
        # Cloud has its own queue (unbounded)
        self.queue = simpy.Store(env)
        self.process = env.process(self.run())

    def run(self):
        while True:
            task = yield self.queue.get()
            # network upload delay is simulated externally in schedule_task (so here we just do compute)
            proc_time = float(task['mi']) / max(1.0, self.mips)
            yield self.env.timeout(proc_time)
            task['finished_at'] = self.env.now
            task['processed_by'] = "cloud"
            SIM_RESULTS.append(task)

# ---------------------------
# Scheduling logic
# ---------------------------
def schedule_task(env, nodes_map, cloud, task, args):
    """
    Decide whether task runs on edge or cloud or gets dropped.
    Offload policies:
      - queue: try edge queue; if full -> offload to cloud
      - always: always offload to cloud (simulate upload + cloud queue)
      - never: try edge, if full -> drop
      - priority-threshold:<mi_thresh> : tasks with mi>thresh -> cloud else try edge
    """
    policy = args.offload_policy
    # if node mapping present, choose target node by hash if node_id absent
    target_node_index = None
    if task.get('node_id'):
        # attempt to map node_id to an integer index deterministically
        nid = task['node_id']
        target_node_index = hash(nid) % args.nodes
    else:
        # simple round-robin or hash by task id
        target_node_index = hash(task['id']) % args.nodes

    node = nodes_map[target_node_index]

    def node_has_space(n):
        return n.queue_len() < n.queue_capacity()

    # policy handling
    if policy.startswith('priority-threshold'):
        parts = policy.split(':')
        thresh = float(parts[1]) if len(parts) > 1 else args.priority_threshold
        if task['mi'] > thresh:
            chosen = 'cloud'
        else:
            chosen = 'edge'
    elif policy == 'always':
        chosen = 'cloud'
    elif policy == 'never':
        chosen = 'edge'
    else:  # default "queue"
        # try edge first
        if node_has_space(node):
            chosen = 'edge'
        else:
            chosen = 'cloud'

    # perform chosen action
    if chosen == 'edge':
        if node_has_space(node):
            node.enqueue(task)
            task['queued_at'] = env.now
            task['assigned_to'] = f"edge-{node.node_index}"
            return 'enqueued'
        else:
            # edge full
            if policy == 'never':
                task['dropped_at'] = env.now
                task['assigned_to'] = 'dropped'
                SIM_RESULTS.append(task)
                return 'dropped'
            else:
                # fallback to cloud
                chosen = 'cloud'

    if chosen == 'cloud':
        # simulate upload latency: data_mb / bandwidth_mbps (MB * 8 / Mbps)
        if cloud is None:
            # no cloud configured -> drop
            task['dropped_at'] = env.now
            task['assigned_to'] = 'dropped-no-cloud'
            SIM_RESULTS.append(task)
            return 'dropped-no-cloud'
        upload_time = 0.0
        try:
            upload_time = (float(task.get('data_mb', 0.0)) * 8.0) / max(1e-6, cloud.bandwidth_mbps)
        except Exception:
            upload_time = 0.0
        # Simulate upload (non-blocking to other scheduling; schedule a process to delay then push to cloud queue)
        def _send_to_cloud(env, cloud, task, upl_time):
            yield env.timeout(upl_time)
            # mark queued for cloud
            task['queued_at'] = env.now
            task['assigned_to'] = 'cloud'
            cloud.queue.put(task)

        env.process(_send_to_cloud(env, cloud, task, upload_time))
        return 'offloaded'

# ---------------------------
# Top-level simulation
# ---------------------------
def run_simulation(args):
    # load tasks
    tasks = load_tasks(args.tasks)
    if len(tasks) == 0:
        print("No tasks found in tasks CSV.")
        return

    # Create environment
    env = simpy.Environment()
    # Create nodes
    nodes_map = {}
    for i in range(args.nodes):
        nodes_map[i] = EdgeNode(env, i, args.edge_mips, args.max_queue)

    # Create cloud
    cloud = Cloud(env, args.cloud_mips, args.bandwidth_mbps) if args.cloud_mips and args.bandwidth_mbps else None

    # schedule generator: at each task arrival, schedule according to policy
    def task_generator(env, tasks):
        for t in tasks:
            # wait until arrival
            if t['arrival'] > env.now:
                yield env.timeout(t['arrival'] - env.now)
            # annotate arrival
            task = dict(t)  # shallow copy
            task['created_at'] = env.now
            # schedule
            res = schedule_task(env, nodes_map, cloud, task, args)
            task['scheduling_decision'] = res
        # No more tasks
        return

    # start generator
    env.process(task_generator(env, tasks))

    # run
    print(f"Starting sim for {args.sim_time} seconds with {args.nodes} nodes...")
    env.run(until=args.sim_time)
    print("Simulation finished.")

# ---------------------------
# Results & plotting
# ---------------------------
def write_results_and_plots(out_log):
    if len(SIM_RESULTS) == 0:
        print("No results to write.")
        return
    # normalize results into DataFrame
    rows = []
    for r in SIM_RESULTS:
        started = r.get('started_at', None)
        finished = r.get('finished_at', None)
        queued = r.get('queued_at', None)
        latency = None
        if started is not None and finished is not None:
            latency = finished - r.get('created_at', started)
        rows.append({
            'task_id': r.get('id'),
            'created_at': r.get('created_at'),
            'queued_at': queued,
            'started_at': started,
            'finished_at': finished,
            'processed_by': r.get('processed_by', r.get('assigned_to')),
            'scheduling_decision': r.get('scheduling_decision'),
            'mi': r.get('mi'),
            'data_mb': r.get('data_mb', 0.0),
            'latency_s': latency
        })

    df = pd.DataFrame(rows)
    # write CSV
    df.to_csv(out_log, index=False)
    print(f"Wrote results to {out_log}")

    # plot latency histogram (only for finished tasks)
    lat = df['latency_s'].dropna()
    if len(lat) > 0:
        plt.figure()
        plt.hist(lat, bins=30)
        plt.xlabel('Latency (s)')
        plt.ylabel('Count')
        plt.title('Latency distribution')
        plt.tight_layout()
        plt.savefig('latency_hist.png')
        print("Saved latency_hist.png")
    else:
        print("No latency data to plot.")

    # ratio processed by edge/cloud/dropped
    by = df['processed_by'].fillna(df['scheduling_decision']).fillna('unknown')
    counts = by.value_counts()
    plt.figure()
    counts.plot.pie(autopct='%1.1f%%')
    plt.ylabel('')
    plt.title('Processed by (edge / cloud / dropped)')
    plt.tight_layout()
    plt.savefig('ratios.png')
    print("Saved ratios.png")


# ---------------------------
# Global container for results
# ---------------------------
SIM_RESULTS = []

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tasks', required=True, help='Path to tasks CSV')
    p.add_argument('--nodes', type=int, default=1, help='Number of edge nodes')
    p.add_argument('--edge-mips', type=float, default=800.0, help='Edge node MIPS (MI per second)')
    p.add_argument('--cloud-mips', type=float, default=5000.0, help='Cloud MIPS')
    p.add_argument('--bandwidth_mbps', type=float, default=8.0, help='Network bandwidth (Mbps) used for upload time calc')
    p.add_argument('--max-queue', type=int, default=4, help='Per-edge-node queue capacity (use 0 or -1 for infinite)')
    p.add_argument('--sim-time', type=float, default=600.0, help='Sim time (seconds)')
    p.add_argument('--offload-policy', default='queue',
                   help='Offload policy: queue | always | never | priority-threshold[:MI]')
    p.add_argument('--priority-threshold', type=float, default=2000.0,
                   help='MI threshold used by priority-threshold policy (if chosen)')
    p.add_argument('--out-log', default='results.csv', help='Output CSV path')
    return p.parse_args()

# ---------------------------
# Entry point
# ---------------------------
if __name__ == '__main__':
    args = parse_args()

    # normalize max_queue: allow -1 or 0 for infinite
    if args.max_queue is not None and args.max_queue <= 0:
        args.max_queue = None

    if not os.path.exists(args.tasks):
        print(f"Tasks file not found: {args.tasks}")
        sys.exit(1)

    # Run simulation
    run_simulation(args)

    # Write results and plots
    write_results_and_plots(args.out_log)

    print("Done.")
