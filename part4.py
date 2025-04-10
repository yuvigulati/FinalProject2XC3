# === PART 4: A* ALGORITHM ===

import math
import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
import os

# Haversine distance used for heuristics
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Heuristic Functions
def generate_heuristic(destination_id, coords):
    lat_dest, lon_dest = coords[destination_id]
    return {
        node: haversine(lat, lon, lat_dest, lon_dest)
        for node, (lat, lon) in coords.items()
    }

def zero_heuristic(destination_id, coords):
    return {node: 0 for node in coords}

def random_heuristic(destination_id, coords):
    return {node: random.uniform(0, 10000) for node in coords}

# A* Algorithm
# === A* ALGORITHM (FIXED FOR PART 3.1) ===
def A_Star(graph, source, destination, heuristic):
    g_scores = {node: float('inf') for node in graph}
    f_scores = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    g_scores[source] = 0
    f_scores[source] = heuristic[source]

    visited = set()
    queue = [(f_scores[source], source)]

    while queue:
        _, u = heapq.heappop(queue)
        if u == destination:
            break
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph[u].items():
            tentative_g = g_scores[u] + weight
            if tentative_g < g_scores[v]:
                prev[v] = u
                g_scores[v] = tentative_g
                f_scores[v] = tentative_g + heuristic[v]
                heapq.heappush(queue, (f_scores[v], v))

    path = reconstruct_path(prev, source, destination)
    return prev, path


# Path reconstruction
def reconstruct_path(prev, source, destination):
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path[0] == source else []

# Visualization for A* quality

def create_heuristic_quality_comparison():
    labels = ["Good Heuristic", "Zero Heuristic", "Random Heuristic"]
    nodes_visited = [46, 267, 278]  # A* only
    execution_times = [0.00012, 0.00025, 0.00029]  # A* times in seconds
    path_lengths = [21001.62, 21001.62, 22679.65]  # in meters
    colors = ['green', 'orange', 'red']

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Heuristic Quality Impact on A* (Baker Street → Becontree)")

    axs[0].bar(labels, nodes_visited, color=colors)
    axs[0].set_title("Nodes Visited")
    axs[0].set_ylabel("Nodes Visited")

    axs[1].bar(labels, [t * 1000 for t in execution_times], color=colors)
    axs[1].set_title("Execution Time")
    axs[1].set_ylabel("Time (ms)")

    axs[2].bar(labels, path_lengths, color=colors)
    axs[2].axhline(y=21001.62, color='gray', linestyle='--', label='Optimal')
    axs[2].set_title("Path Length")
    axs[2].set_ylabel("Length (m)")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("heuristic_quality_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved to {os.path.join(os.getcwd(), 'heuristic_quality_comparison.png')}")
