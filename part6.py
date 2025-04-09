from abc import ABC, abstractmethod
import heapq
import math
import pandas as pd
from collections import defaultdict

# ================================
# Graph Class
# ================================
class Graph:
    def __init__(self):
        self.edges = defaultdict(dict)

    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight
        self.edges[v][u] = weight  # Assume undirected graph

    def neighbors(self, node):
        return self.edges[node].items()

    def get_nodes(self):
        return list(self.edges.keys())

# ================================
# SPAlgorithm Interface
# ================================
class SPAlgorithm(ABC):
    @abstractmethod
    def shortest_path(self, graph, source, destination):
        pass

# ================================
# Dijkstra Algorithm
# ================================
class Dijkstra(SPAlgorithm):
    def shortest_path(self, graph, source, destination):
        dist = {node: float('inf') for node in graph.get_nodes()}
        prev = {node: None for node in graph.get_nodes()}
        dist[source] = 0
        visited = set()
        queue = [(0, source)]

        while queue:
            d, u = heapq.heappop(queue)
            if u in visited:
                continue
            visited.add(u)
            for v, weight in graph.neighbors(u):
                alt = dist[u] + weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(queue, (alt, v))

        return dist, prev

# ================================
# A* Algorithm
# ================================
class A_Star(SPAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def shortest_path(self, graph, source, destination):
        g_scores = {node: float('inf') for node in graph.get_nodes()}
        f_scores = {node: float('inf') for node in graph.get_nodes()}
        prev = {node: None for node in graph.get_nodes()}
        g_scores[source] = 0
        f_scores[source] = self.heuristic[source]
        queue = [(f_scores[source], source)]
        visited = set()

        while queue:
            _, u = heapq.heappop(queue)
            if u == destination:
                break
            if u in visited:
                continue
            visited.add(u)

            for v, weight in graph.neighbors(u):
                tentative_g = g_scores[u] + weight
                if tentative_g < g_scores[v]:
                    prev[v] = u
                    g_scores[v] = tentative_g
                    f_scores[v] = tentative_g + self.heuristic[v]
                    heapq.heappush(queue, (f_scores[v], v))

        return g_scores, prev

# ================================
# Bellman-Ford Algorithm
# ================================
class BellmanFord(SPAlgorithm):
    def shortest_path(self, graph, source, destination):
        dist = {node: float('inf') for node in graph.get_nodes()}
        prev = {node: None for node in graph.get_nodes()}
        dist[source] = 0

        for _ in range(len(graph.get_nodes()) - 1):
            for u in graph.get_nodes():
                for v, weight in graph.neighbors(u):
                    if dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        prev[v] = u

        return dist, prev

# ================================
# ShortPathFinder Adapter
# ================================
class ShortPathFinder:
    def __init__(self, algorithm: SPAlgorithm):
        self.algorithm = algorithm

    def find_path(self, graph, source, destination):
        dist, prev = self.algorithm.shortest_path(graph, source, destination)
        return self.reconstruct_path(prev, source, destination), dist[destination]

    def reconstruct_path(self, prev, source, destination):
        path = []
        node = destination
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path if path[0] == source else []

# ================================
# Utility Functions
# ================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_heuristic(destination_id, coords):
    lat_dest, lon_dest = coords[destination_id]
    return {
        node: haversine(lat, lon, lat_dest, lon_dest)
        for node, (lat, lon) in coords.items()
    }

# ================================
# Example Usage
# ================================
if __name__ == "__main__":
    # Load data
    stations_df = pd.read_csv("london_stations.csv")
    connections_df = pd.read_csv("london_connections.csv")

    coords = {row["id"]: (row["latitude"], row["longitude"]) for _, row in stations_df.iterrows()}

    # Build graph
    graph = Graph()
    for _, row in connections_df.iterrows():
        s1, s2 = row['station1'], row['station2']
        lat1, lon1 = coords[s1]
        lat2, lon2 = coords[s2]
        dist = haversine(lat1, lon1, lat2, lon2)
        graph.add_edge(s1, s2, dist)

    # Define source and destination
    source, destination = 11, 95

    # Dijkstra
    dijkstra_finder = ShortPathFinder(Dijkstra())
    path, distance = dijkstra_finder.find_path(graph, source, destination)
    print("Dijkstra Path:", path)
    print("Distance:", distance)

    # A*
    heuristic = generate_heuristic(destination, coords)
    a_star_finder = ShortPathFinder(A_Star(heuristic))
    path, distance = a_star_finder.find_path(graph, source, destination)
    print("A* Path:", path)
    print("Distance:", distance)

    # Bellman-Ford
    bellman_finder = ShortPathFinder(BellmanFord())
    path, distance = bellman_finder.find_path(graph, source, destination)
    print("Bellman-Ford Path:", path)
    print("Distance:", distance)
