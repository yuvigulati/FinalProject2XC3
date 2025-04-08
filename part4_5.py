# Part 4: A* Algorithm Analysis and Implementation
## Part 4.1: A* Algorithm Implementation
import heapq
import math

def A_Star(graph, source, destination, heuristic):
    # Initialize distances with infinity and predecessors as None
    distances = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    distances[source] = 0

    # Priority queue: (f_score, node)
    # f_score = g_score (distance from start) + h_score (heuristic to goal)
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(source), source))

    # g_scores (actual distances from start)
    g_scores = {node: float('inf') for node in graph}
    g_scores[source] = 0

    while open_set:
        current_f, current_node = heapq.heappop(open_set)

        if current_node == destination:
            break  # Found the destination

        for neighbor, weight in graph[current_node].items():
            # Calculate tentative g_score
            tentative_g = g_scores[current_node] + weight

            if tentative_g < g_scores[neighbor]:
                # This path to neighbor is better than any previous one
                predecessors[neighbor] = current_node
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score, neighbor))

    # Reconstruct the path
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()

    return (predecessors, path) if path[0] == source else (predecessors, [])

#part 5
def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

