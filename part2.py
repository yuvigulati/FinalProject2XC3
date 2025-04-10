import random
import heapq
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


class WeightedGraph:
    #Using a dictionary for the adjacency list:
    #Each key is a node; its value is a list of tuples (neighbor, weight)
    def __init__(self, nodes):
        self.nodes = nodes
        self.adj = {i: [] for i in range(nodes)}
    
    #Add an edge to the graph
    def add_edge(self, src, dst, weight, undirected=True):
        self.adj[src].append((dst, weight))
        if undirected:
            self.adj[dst].append((src, weight))

    #Check if an edge exists between two nodes
    def has_edge(self, src, dst):
        return src in self.graph[dst]
    

#Helper functions
def create_random_weighted_graph(nodes, density, allow_negative=False):
    graph = WeightedGraph(nodes)
    for i in range(nodes):
        for j in range(i + 1, nodes):
            if allow_negative:
                #Choose a random weight from -20 to 20 excluding 0.
                possible_weights = [w for w in range(-20, 21) if w != 0]
                weight = random.choice(possible_weights)
            else:
                weight = random.randint(1, 20)
            graph.add_edge(i, j, weight, undirected=False)
    return graph

def get_accuracy_percentage(baseline, computed):
    accuracies = []
    for node in baseline:
        b = baseline[node]
        c = computed[node]
        if b == float('inf'):
            #If baseline is not reachable then consider accuracy as 100 if computed is also infinite.
            acc = 100 if c == float('inf') else 0
        elif b == 0:
            acc = 100 if c == 0 else 0
        else:
            error_percentage = abs(c - b) / abs(b) * 100 #Calculate the error percentage
            acc = max(0, 100 - error_percentage) #Accuracy is 100% - error percentage
        accuracies.append(acc)
    return np.mean(accuracies)

#Part 2.1: Dijkstra's Algorithm
def dijkstra(graph, source, k):
    n = graph.nodes #No. of nodes in the graph

    distance = {i: float('inf') for i in range(n)} #Initialize distances to infinity
    distance[source] = 0 #Distance to source is 0

    path = {i: [] for i in range(n)} #Initialize paths
    path[source] = [source] #Path to source is just the source itself

    relax_count = {i: 0 for i in range(n)} #Relaxation counter for each node

    heap = [(0, source)] #Priority queue for Dijkstra's algorithm
    
    while heap:
        dist_u, u = heapq.heappop(heap) #Get the node with the smallest distance

        if dist_u > distance[u]: #If the distance is not optimal, skip it
            continue
        
        for v, weight in graph.adj[u]: #For each neighbor of u
            new_dist = distance[u] + weight
            if new_dist < distance[v] and relax_count[v] < k: #Relax edge if the new computed distance is lower and node v hasn't exceeded k relaxations
                distance[v] = new_dist
                path[v] = path[u] + [v]
                relax_count[v] += 1
                heapq.heappush(heap, (new_dist, v))
    
    return distance, path #Return the distances and paths from the source to all other nodes


#Part 2.2: Bellman-Ford Algorithm
def bellman_ford(graph, source, k):
    n = graph.nodes #No. of nodes in the graph

    distance = {i: float('inf') for i in range(n)} #Initialize distances to infinity
    distance[source] = 0 #Distance to source is 0 

    path = {i: [] for i in range(n)} #Initialize paths
    path[source] = [source] #Path to source is just the source itself

    relax_count = {i: 0 for i in range(n)} #Relaxation counter for each node
    
    for i in range(k):
        updated = False
        for u in range(n): #For each node in the graph
            if distance[u] == float('inf'): #If the node is unreachable, skip it
                continue
            for v, weight in graph.adj[u]: #For each neighbor of u
                new_dist = distance[u] + weight 
                if new_dist < distance[v] and relax_count[v] < k: #Relax edge if the new computed distance is lower and node v hasn't exceeded k relaxations
                    distance[v] = new_dist
                    path[v] = path[u] + [v]
                    relax_count[v] += 1
                    updated = True
        if not updated:
            break

    return distance, path #Return the distances and paths from the source to all other nodes

#Part 2.3: Experiment Comparison Function
def experiment_comparison(alg_fn, alg_name, allow_negative=False, save_path=None):
    # Parameters for experiments:
    nodes_list = [30, 60, 90]           # Different graph sizes.
    densities = [0.1, 0.5, 0.99]           # Different edge probabilities.
    k_values = [1, 15, 29]                # Different maximum relaxations.
    trials = 10                         # Number of trials per combination.
    
    # results[nodes][k][density] = {"time": avg_time, "accuracy": avg_accuracy, "memory": avg_memory}
    results = {}
    if allow_negative:
        for nodes in nodes_list:
            results[nodes] = {}
            for k in k_values:
                results[nodes][k] = {}
                for density in densities:
                    time_vals = []
                    acc_vals = []
                    mem_vals = []
                    for _ in range(trials):
                        # Create random graph with negative weights allowed.
                        graph = create_random_weighted_graph(nodes, density, allow_negative=True)
                        
                        # Measure execution time and memory using tracemalloc.
                        tracemalloc.start()
                        start_time = time.time()
                        res = alg_fn(graph, 0, k)  # Run algorithm with given k.
                        cur_mem, peak_mem = tracemalloc.get_traced_memory()
                        exec_time = time.time() - start_time
                        tracemalloc.stop()
                        
                        time_vals.append(exec_time)
                        mem_vals.append(peak_mem / 1024.0)  # Convert bytes to kilobytes.
                        
                        # Compute accuracy percentage using a baseline (k = nodes - 1)
                        baseline = alg_fn(graph, 0, nodes - 1)
                        baseline_distance = baseline[0]  # Extract distance dictionary.
                        res_distance = res[0]            # Extract distance dictionary.
                        acc_percentage = get_accuracy_percentage(baseline_distance, res_distance)
                        acc_vals.append(acc_percentage)
                    
                    avg_time = sum(time_vals) / trials
                    avg_mem = sum(mem_vals) / trials
                    avg_acc = sum(acc_vals) / trials
                    results[nodes][k][density] = {"time": avg_time, "accuracy": avg_acc, "memory": avg_mem}
    else:
        for nodes in nodes_list:
            results[nodes] = {}
            for k in k_values:
                results[nodes][k] = {}
                for density in densities:
                    time_vals = []
                    acc_vals = []
                    mem_vals = []
                    for _ in range(trials):
                        # Create random graph with only nonnegative weights.
                        graph = create_random_weighted_graph(nodes, density, allow_negative=False)
                        
                        # Measure execution time and memory using tracemalloc.
                        tracemalloc.start()
                        start_time = time.time()
                        res = alg_fn(graph, 0, k)  # Run algorithm with given k.
                        cur_mem, peak_mem = tracemalloc.get_traced_memory()
                        exec_time = time.time() - start_time
                        tracemalloc.stop()
                        
                        time_vals.append(exec_time)
                        mem_vals.append(peak_mem / 1024.0)  # Convert bytes to kilobytes.
                        
                        # Baseline using unconstrained relaxations (k = nodes - 1)
                        baseline = alg_fn(graph, 0, nodes - 1)
                        baseline_distance = baseline[0]  # Extract distance dictionary.
                        res_distance = res[0]            # Extract distance dictionary.
                        acc_percentage = get_accuracy_percentage(baseline_distance, res_distance)
                        acc_vals.append(acc_percentage)
                    
                    avg_time = sum(time_vals) / trials
                    avg_mem = sum(mem_vals) / trials
                    avg_acc = sum(acc_vals) / trials
                    results[nodes][k][density] = {"time": avg_time, "accuracy": avg_acc, "memory": avg_mem}
    
    # --- Plotting: Merged Bar Graphs for All Nodal Iterations ---
    # Build categories of the form "Nodes X, Den Y"
    categories = []
    for nodes in nodes_list:
        for density in densities:
            categories.append(f"Nodes {nodes}, Den {density}")
    x = np.arange(len(categories))
    bar_width = 0.2  # Width for each bar
    
    # Determine the number of subplots:
    # Always plot three subplots (Time, Accuracy, Memory) with appropriate headers.
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
    ax_time, ax_acc, ax_mem = axes
    
    # Plot Execution Time.
    for i, k in enumerate(k_values):
        time_vals = []
        for nodes in nodes_list:
            for density in densities:
                time_vals.append(results[nodes][k][density]["time"])
        ax_time.bar(x + i * bar_width, time_vals, width=bar_width, label=f"k = {k}")
    ax_time.set_xlabel("Graph Category (Nodes, Density)")
    ax_time.set_ylabel("Avg Time (s)")
    ax_time.set_title(f"{alg_name} - Time Performance (Graphs are {'Negative' if allow_negative else 'Positive'})")
    ax_time.set_xticks(x + bar_width * (len(k_values) - 1) / 2)
    ax_time.set_xticklabels(categories, rotation=45, ha="right")
    ax_time.legend()
    
    # Plot Accuracy.
    for i, k in enumerate(k_values):
        acc_vals = []
        for nodes in nodes_list:
            for density in densities:
                acc_vals.append(results[nodes][k][density]["accuracy"])
        ax_acc.bar(x + i * bar_width, acc_vals, width=bar_width, label=f"k = {k}")
    ax_acc.set_xlabel("Graph Category (Nodes, Density)")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(f"{alg_name} - Accuracy Performance (Graphs are {'Negative' if allow_negative else 'Positive'})")
    ax_acc.set_xticks(x + bar_width * (len(k_values) - 1) / 2)
    ax_acc.set_xticklabels(categories, rotation=45, ha="right")
    ax_acc.legend()
    
    # Plot Memory usage.
    for i, k in enumerate(k_values):
        mem_vals = []
        for nodes in nodes_list:
            for density in densities:
                mem_vals.append(results[nodes][k][density]["memory"])
        ax_mem.bar(x + i * bar_width, mem_vals, width=bar_width, label=f"k = {k}")
    ax_mem.set_xlabel("Graph Category (Nodes, Density)")
    ax_mem.set_ylabel("Avg Memory (KB)")
    ax_mem.set_title(f"{alg_name} - Memory Performance (Graphs are {'Negative' if allow_negative else 'Positive'})")
    ax_mem.set_xticks(x + bar_width * (len(k_values) - 1) / 2)
    ax_mem.set_xticklabels(categories, rotation=45, ha="right")
    ax_mem.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def run_experiment_via_input():
    print("Select the algorithm to run the experiment:")
    print("1: Modified Dijkstra's Algorithm (nonnegative weights only)")
    print("2: Modified Bellman-Ford Algorithm")
    algo_choice = input("Enter your choice (1 or 2): ").strip()

    if algo_choice == "1":
        print("Running experiment for Modified Dijkstra's Algorithm (nonnegative weights only)...")
        experiment_comparison(dijkstra, "Modified Dijkstra", allow_negative=False, save_path="comparison_dijkstra.png")
    elif algo_choice == "2":
        print("For Modified Bellman-Ford Algorithm, choose the weight type:")
        print("1: Positive weights only")
        print("2: Negative weights allowed")
        weight_choice = input("Enter your choice (1 or 2): ").strip()
        if weight_choice == "1":
            print("Running experiment for Modified Bellman-Ford Algorithm with positive weights only...")
            experiment_comparison(bellman_ford, "Modified Bellman-Ford", allow_negative=False, save_path="comparison_bf_positive.png")
        elif weight_choice == "2":
            print("Running experiment for Modified Bellman-Ford Algorithm with negative weights allowed...")
            experiment_comparison(bellman_ford, "Modified Bellman-Ford", allow_negative=True, save_path="comparison_bf_negative.png")
        else:
            print("Invalid input for weight type. Exiting experiment.")
    else:
        print("Invalid algorithm selection. Please enter either 1 or 2.")

if __name__ == "__main__":
    run_experiment_via_input()