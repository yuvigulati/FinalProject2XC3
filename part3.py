import math

def floyd_warshall(graph):
    V = len(graph)
    dist = [[float('inf')] * V for _ in range(V)]
    prev = [[None] * V for _ in range(V)]
    
    # Initialize
    for i in range(V):
        dist[i][i] = 0
        for j in graph[i]:
            dist[i][j] = graph[i][j]
            prev[i][j] = i
    
    # Relax edges
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]
    
    # Check for negative cycles
    for i in range(V):
        if dist[i][i] < 0:
            return None, None, "Negative cycle detected"
    
    return dist, prev, None
