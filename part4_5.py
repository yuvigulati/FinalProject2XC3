
#  IMPORTS – Used in both Part 4 & Part 5

import pandas as pd
import math
import heapq
import time
import random
from collections import defaultdict


#  DATA LOADING – Part 5

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


#  HAVERSINE DISTANCE FUNCTION – Part 5 (used in edge weights + heuristic)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


#  GRAPH CONSTRUCTION – Part 5

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


# DIJKSTRA'S ALGORITHM – Part 5

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


#  A* ALGORITHM – Part 4 & Part 5

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
    return g_scores, prev, visited


#  HEURISTIC FUNCTIONS – Part 4

def generate_heuristic(destination_id, coords):  # Good heuristic
    lat_dest, lon_dest = coords[destination_id]
    return {
        node: haversine(lat, lon, lat_dest, lon_dest)
        for node, (lat, lon) in coords.items()
    }

def zero_heuristic(destination_id, coords):  # A* behaves like Dijkstra
    return {node: 0 for node in coords}

def random_heuristic(destination_id, coords):  # Bad heuristic
    return {node: random.uniform(0, 10000) for node in coords}


#  PATH & LINE COUNT HELPERS – Part 5

def reconstruct_path(prev, source, destination):
    path = []
    node = destination
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path if path[0] == source else []

def count_lines(path, line_data):
    if len(path) < 2:
        return 0
    used_lines = set()
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        used_lines.update(line_data[u][v])
    return len(used_lines)


#  ALGORITHM COMPARISON – Part 5

def compare_algorithms(graph, coords, line_data, test_pairs, heuristic_func=generate_heuristic):
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

#  RUNNING TEST CASES – Part 5 (also used for Part 4 heuristic testing)

graph, line_data = build_graph(connections_df, station_coords)

test_pairs = [
    (11, 21),     # Same line
    (11, 47),     # Adjacent lines
    (11, 95),     # Multi-transfer trip
    (58, 122),    # Random
    (74, 144)     # Edge-to-edge
]

print("=== A* vs Dijkstra using GOOD heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs)

print("\n=== A* vs Dijkstra using ZERO heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs, heuristic_func=zero_heuristic)

print("\n=== A* vs Dijkstra using RANDOM heuristic ===")
compare_algorithms(graph, station_coords, line_data, test_pairs, heuristic_func=random_heuristic)
