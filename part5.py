# === IMPORTS ===
import pandas as pd
import math
import heapq
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os


# === DATA LOADING ===
stations_df = pd.read_csv("london_stations.csv")
connections_df = pd.read_csv("london_connections.csv")

station_coords = {
    row["id"]: (row["latitude"], row["longitude"])
    for _, row in stations_df.iterrows()
}
station_names = {
    row["id"]: row["name"]
    for _, row in stations_df.iterrows()
}


# === HAVERSINE DISTANCE FUNCTION ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# === GRAPH CONSTRUCTION ===
def build_graph(connections, coords):
    graph = defaultdict(dict)
    connection_lines = defaultdict(lambda: defaultdict(list))
    for _, row in connections.iterrows():
        s1, s2, line = row['station1'], row['station2'], row['line']
        lat1, lon1 = coords[s1]
        lat2, lon2 = coords[s2]
        dist = haversine(lat1, lon1, lat2, lon2)
        graph[s1][s2] = dist
        graph[s2][s1] = dist
        connection_lines[s1][s2].append(line)
        connection_lines[s2][s1].append(line)
    return graph, connection_lines


# === DIJKSTRA'S ALGORITHM ===
def dijkstra(graph, source):
    dist = {node: float('inf') for node in graph}
    prev = {node: None for node in graph}
    dist[source] = 0
    visited = set()
    queue = [(0, source)]

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph[u].items():
            alt = dist[u] + weight
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(queue, (alt, v))
    return dist, prev, visited


# === RECONSTRUCT PATH ===
def reconstruct_path(prev, source, destination):
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path[0] == source else []


# === COUNT LINES USED IN PATH ===
def count_lines(path, line_data):
    if len(path) < 2:
        return 0
    used_lines = set()
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        used_lines.update(line_data[u][v])
    return len(used_lines)


# === COMPARE A* vs DIJKSTRA ===
def compare_algorithms(graph, coords, line_data, test_pairs, heuristic_func):
    for source, destination in test_pairs:
        print(f"\nFrom {station_names[source]} to {station_names[destination]}")
        heuristic = heuristic_func(destination, coords)

        # Dijkstra
        t0 = time.time()
        d_dist, d_prev, d_visited = dijkstra(graph, source)
        d_time = time.time() - t0
        d_path = reconstruct_path(d_prev, source, destination)

        # A*
        t0 = time.time()
        a_dist, a_prev, a_visited = A_Star(graph, source, destination, heuristic)
        a_time = time.time() - t0
        a_path = reconstruct_path(a_prev, source, destination)

        # Results
        print(f"Dijkstra: time = {d_time:.5f}s, nodes = {len(d_visited)}, path = {d_dist[destination]:.2f} m, lines used = {count_lines(d_path, line_data)}")
        print(f"A*      : time = {a_time:.5f}s, nodes = {len(a_visited)}, path = {a_dist[destination]:.2f} m, lines used = {count_lines(a_path, line_data)}")


# === TEST CASES ===
graph, line_data = build_graph(connections_df, station_coords)
test_pairs = [(11, 21), (11, 47), (11, 95), (58, 122), (74, 144)]

print("=== A* vs Dijkstra using GOOD heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs, generate_heuristic)

print("\n=== A* vs Dijkstra using ZERO heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs, zero_heuristic)

print("\n=== A* vs Dijkstra using RANDOM heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs, random_heuristic)


# === PERFORMANCE VISUALIZATION ===
def create_visualizations():
    good_heuristic = [
        ("Baker St → Becontree", 302, 46, 0.00025, 0.00012, 4),
        ("Baker St → Chalk Farm", 302, 19, 0.00023, 0.00009, 4),
        ("Baker St → Finsbury Park", 302, 9, 0.00025, 0.00008, 4),
        ("Colindale → Kensington", 302, 24, 0.00023, 0.00009, 5),
        ("Earl's Court → Kingsbury", 302, 74, 0.00022, 0.00012, 5)
    ]
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({'font.size': 12, 'figure.titlesize': 14, 'axes.labelsize': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('A* vs Dijkstra Performance (Good Heuristic)')
    routes = [x[0] for x in good_heuristic]
    x = np.arange(len(routes))
    width = 0.35
    ax1.bar(x - width/2, [x[1] for x in good_heuristic], width, label='Dijkstra', color='navy')
    ax1.bar(x + width/2, [x[2] for x in good_heuristic], width, label='A*', color='darkorange')
    ax1.set_ylabel('Nodes Visited')
    ax1.set_title('Node Exploration Efficiency')
    ax1.set_xticks(x)
    ax1.set_xticklabels(routes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar(x - width/2, [x[3]*1000 for x in good_heuristic], width, label='Dijkstra', color='navy')
    ax2.bar(x + width/2, [x[4]*1000 for x in good_heuristic], width, label='A*', color='darkorange')
    ax2.set_ylabel('Time (milliseconds)')
    ax2.set_title('Execution Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(routes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_heuristic_quality_comparison():
    labels = ["Good Heuristic", "Zero Heuristic", "Random Heuristic"]
    nodes_visited = [46, 267, 278]
    execution_times = [0.00012, 0.00025, 0.00029]
    path_lengths = [21001.62, 21001.62, 22679.65]
    colors = ['green', 'orange', 'red']
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Heuristic Quality Impact on A* (Baker Street → Becontree)")
    axs[0].bar(labels, nodes_visited, color=colors)
    axs[0].set_title("Nodes Visited Comparison")
    axs[0].set_ylabel("Nodes Visited")
    axs[1].bar(labels, [t * 1000 for t in execution_times], color=colors)
    axs[1].set_title("Execution Time Comparison")
    axs[1].set_ylabel("Time (milliseconds)")
    axs[2].bar(labels, path_lengths, color=colors)
    axs[2].axhline(y=21001.62, color='gray', linestyle='--', label='Optimal')
    axs[2].set_title("Path Optimality Comparison")
    axs[2].set_ylabel("Path Length (meters)")
    axs[2].legend()
    plt.tight_layout()
    plt.savefig("heuristic_quality_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


create_visualizations()
create_heuristic_quality_comparison()