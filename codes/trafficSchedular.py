import logging
import yafs
from yafs.application import Application, Message
from yafs.topology import Topology
from yafs.placement import Placement
from yafs.selection import Selection
from yafs.distribution import deterministic_distribution

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------
# Custom Priority Selection
# ---------------------------------------------------------
class PrioritySelection(Selection):
    def __init__(self, priorities, deadlines):
        super().__init__("PrioritySelection")
        self.priorities = priorities
        self.deadlines = deadlines

    def get_path(self, sim, app_name, message, topology_src, alloc_DES):
        # Priority-based scheduling demo (simplified)
        service = message.dst
        priority = self.priorities.get(message.name, "LOW")
        deadline = self.deadlines.get(message.name, 999)

        path = sim.topology.get_shortest_path(topology_src, alloc_DES[0])
        logging.info(f"[PRIORITY={priority}, Deadline={deadline}] Message {message.name} "
                     f"from {message.src} → {service} via {path}")
        return [path]

# ---------------------------------------------------------
# Custom Placement (static mapping)
# ---------------------------------------------------------
class TrafficPlacement(Placement):
    def initial_allocation(self, sim, app_name, app_module):
        mapping = {
            "EmergencyService": [0],   # edge
            "CongestionService": [0],  # edge
            "SpeedService": [0],       # edge
            "CountingService": [1]     # cloud
        }
        return mapping.get(app_module, [])

# ---------------------------------------------------------
# Application Model
# ---------------------------------------------------------
def create_app():
    app = Application("TrafficMonitoring")

    m_emergency = Message("Emergency", "Sensor", "EmergencyService", instructions=500, bytes=500)
    m_congestion = Message("Congestion", "Sensor", "CongestionService", instructions=400, bytes=400)
    m_speed = Message("Speed", "Sensor", "SpeedService", instructions=300, bytes=300)
    m_count = Message("Counting", "Sensor", "CountingService", instructions=200, bytes=200)

    app.add_service_module("EmergencyService", m_emergency)
    app.add_service_module("CongestionService", m_congestion)
    app.add_service_module("SpeedService", m_speed)
    app.add_service_module("CountingService", m_count)

    return app

# ---------------------------------------------------------
# Topology
# ---------------------------------------------------------
def create_topology():
    topo = Topology()
    # Two nodes: 0=edge, 1=cloud
    topo.add_node(0, {"name": "EdgeNode", "IPT": 500, "RAM": 1000})
    topo.add_node(1, {"name": "CloudNode", "IPT": 2000, "RAM": 4000})
    # Links
    topo.add_link(0, 1, {"BW": 10, "PR": 2})
    topo.add_link(1, 0, {"BW": 10, "PR": 2})
    return topo

# ---------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------
def main():
    app = create_app()

    priorities = {
        "Emergency": "HIGH",
        "Congestion": "HIGH",
        "Speed": "MEDIUM",
        "Counting": "LOW"
    }
    deadlines = {
        "Emergency": 2.0,
        "Congestion": 3.0,
        "Speed": 5.0,
        "Counting": 8.0
    }

    topo = create_topology()
    placement = TrafficPlacement(name="TrafficPlacement")
    selector = PrioritySelection(priorities, deadlines)

    sim = yafs.core.Sim(topo, default_results_path="results/")
    sim.deploy_app(app, placement, selector)

    dist = deterministic_distribution(name="Deterministic", time=100)

    # Source definitions (dictionary-style for your YAFS version)
    src1 = {"id": 0, "app": app.name, "message": "Emergency", "distribution": dist}
    src2 = {"id": 0, "app": app.name, "message": "Congestion", "distribution": dist}
    src3 = {"id": 0, "app": app.name, "message": "Speed", "distribution": dist}
    src4 = {"id": 0, "app": app.name, "message": "Counting", "distribution": dist}

    sim.deploy_source(src1)
    sim.deploy_source(src2)
    sim.deploy_source(src3)
    sim.deploy_source(src4)

    logging.info("🚦 Starting Traffic Monitoring Simulation...")
    sim.run(until=1000)
    logging.info("✅ Simulation Finished. Check results/ folder.")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
