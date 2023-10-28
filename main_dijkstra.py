

import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
random.seed(0)



class graphvisualization:
    def __init__(self):
        self.G=nx.Graph()
        self.count=0
    
    def addedge(self,a,b,weight):
        self.G.add_edge(a,b,weight=weight)
        self.count=self.count+1
        
    def set_position(self,pos):
        self.pos=pos
        
    def get_weights(self):
        return [data['weight'] for _,_,data in self.G.edges(data=True)]
    
    def split_nodes(self,pairwise_nodes):
        first_nodes = []
        second_nodes = []
        for pair in pairwise_nodes:
            first_nodes.append(pair[0])
            second_nodes.append(pair[1])
        return first_nodes, second_nodes
        
    def pairwise_nodes(self):
        pairwise_nodes = list(self.G.edges)
        nodes=self.G.nodes
        first_nodes, second_nodes =self.split_nodes(pairwise_nodes)
        return first_nodes, second_nodes,nodes
    
    def get_min_distance(self, distances, visited):
        min_distance = float('inf')
        min_node = None
        for node in distances:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node
        return min_node
    
    def create_q_table(self):
        num_nodes = len(self.G)
        self.Q = np.zeros((num_nodes, num_nodes))

        for node in self.G:
            for neighbor in self.G[node]:
                self.Q[node-1][neighbor-1] = 0  # Initialize Q-values to 0 for valid state-action pairs


    def print_q_table(self):
        q_table=self.Q
        return q_table
        
    
    def visualize(self):
        nx.draw_networkx(self.G, pos=self.pos, with_labels=True)
        nx.draw_networkx_edge_labels(self.G, pos=self.pos, edge_labels= nx.get_edge_attributes(self.G,"weight"))
        plt.show()
        
        
        
G=graphvisualization()
G.addedge(0, 2, weight=10)
G.addedge(1, 2, weight=20)
G.addedge(1, 3, weight=22)
G.addedge(5, 3, weight=26)
G.addedge(3, 4, weight=35)
G.addedge(1, 0, weight=55)
G.addedge(3, 0, weight=43)
G.addedge(3, 1, weight=67)
G.addedge(3, 6, weight=150)
G.addedge(4, 7, weight=12)
G.addedge(5, 8, weight=27)
G.addedge(8, 9, weight=33)
G.addedge(9, 10, weight=190)
G.addedge(4, 11, weight=140)
G.addedge(9, 12, weight=15)
G.addedge(10, 13, weight=37)
G.addedge(11, 14, weight=26)
G.addedge(2, 15, weight=18)
G.addedge(7, 16, weight=38)
G.addedge(10, 17, weight=45)
G.addedge(8, 17, weight=56)
G.addedge(11, 18, weight=34)
G.addedge(12, 19, weight=43)
G.addedge(6, 20, weight=52)
G.addedge(7, 21, weight=102)
G.addedge(19, 22, weight=128)
G.addedge(13, 23, weight=49)
G.addedge(20, 24, weight=64)
G.addedge(18, 25, weight=78)


pos=nx.spring_layout(G.G)
G.set_position(pos)
G.visualize()
m=G.count
pairwise_nodes=G.pairwise_nodes()
first_nodes,second_nodes,nodes=G.pairwise_nodes()
G.create_q_table()

Q_1=G.print_q_table()
df1=pd.DataFrame(Q_1)

df1.to_csv('table1.csv', index=False)  # Export as CSV
df1.to_excel('table1.xlsx', index=False)  # Export as Excel

#dist=G.get_weights()
def input_output(G,m):
    #first vertex
    first_vertex=np.array(first_nodes)
    #second vertex
    second_vertex=np.array(second_nodes)
    #distance
    dist_edge=G.get_weights()
    #distance=[random.randint(10,200) for x in range(m-1)]
    dist_edge=[dist_edge[i] for i in range(m-1)]
    print(dist_edge)
    #road condition
    rank_road=[random.randint(1,5) for x in range(m-1)]
    road_cond=[rank_road[i] for i in range(m-1)]
    #road width
    road_width=[random.randint(4,30) for x in range(m-1)]
    width_list=[road_width[i] for i in range(m-1)]
    width_list=np.array(width_list)
    #traffic condition
    road_traffic=[random.randint(1,4) for x in range(m-1)]
    traffic_cond=[road_traffic[i] for i in range(m-1)]
    #distance by width fraction
    dist_width=[dist_edge[i]/width_list[i] for i in range(m-1)]
    #Finding the coefficients by proportion
    constant=1
    total=0
    alpha=[]
    beta=[]
    gamma=[]
    for i in range(m-1):
        total=dist_width[i]+road_cond[i]+traffic_cond[i]
        int(total)
        #find alpha
        alpha.append(dist_width[i]/total)
        #find beta
        beta.append(road_cond[i]/total)
        #find gamma
        gamma.append(road_cond[i]/total)
    print("\nvalues of alpha:\n",alpha)
    print("\nvalues of beta are:\n",beta)
    print("\nvalues of gamma:\n",gamma)
    
    c_new=alpha[0]*dist_width[0]+beta[0]*road_cond[0]+gamma[0]*traffic_cond[0]
    c=[alpha[i]*(dist_width[i])+beta[i]*road_cond[i]+gamma[i]*traffic_cond[i] for i in range(m-1)]
    c=np.array(c)
    print("\nvalues of c are:\n",c)

    return first_vertex,second_vertex,dist_edge,road_cond,width_list,traffic_cond,dist_width,c

first_vertices,second_vertices,distance,road_condition,width,traffic_condition,distance_width,c=input_output(G, m)



df=pd.DataFrame({
                'first_vertices':first_vertices,
                'second_vertices':second_vertices,
                'distance':distance,
                'road_condition':road_condition,
                'width':width,
                'traffic_condition':traffic_condition,
                'distance_width':distance_width,
                'c':c
                })
    
data=df.to_csv('data.csv')
c=np.array(c)
print(c)

G_new = nx.Graph()


import heapq

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def add_edge(self, u, v, weight):
        self.graph[u][v] = weight

    def dijkstra(self, src):
        distances = [float("inf")] * self.V
        distances[src] = 0
        priority_queue = [(0, src)]  # Tuple: (distance, node)
        paths = [[] for _ in range(self.V)]  # Initialize paths

        while priority_queue:
            dist_u, u = heapq.heappop(priority_queue)

            for v in range(self.V):
                if (
                        self.graph[u][v] != 0 and
                        distances[v] > distances[u] + self.graph[u][v]
                ):
                    distances[v] = distances[u] + self.graph[u][v]
                    paths[v] = paths[u] + [u]  # Update the path
                    heapq.heappush(priority_queue, (distances[v], v))

        return distances, paths



# Read the data.csv file to obtain the pairs of nodes and their 'c' values
data = pd.read_csv('data.csv')

# Determine the number of nodes based on the unique values in 'first_vertices' and 'second_vertices' columns
num_nodes = len(set(data['first_vertices']).union(set(data['second_vertices'])))

# Initialize an empty adjacency matrix filled with zeros
adjacency_matrix = np.zeros((num_nodes, num_nodes))

# Fill the adjacency matrix with 'c' values based on the pairs of nodes
for _, row in data.iterrows():
    first_node = int(row['first_vertices'])
    second_node = int(row['second_vertices'])
    c_value = row['c']
    
    # Assign the 'c' value to the corresponding entry in the adjacency matrix
    adjacency_matrix[first_node, second_node] = c_value
    adjacency_matrix[second_node, first_node] = c_value  # Since the graph is undirected

# Print the adjacency matrix
print("Adjacency Matrix with 'c' values:")
print(adjacency_matrix)

g = Graph(len(adjacency_matrix))
for u in range(len(adjacency_matrix)):
    for v in range(len(adjacency_matrix)):
        if adjacency_matrix[u][v] > 0:
            g.add_edge(u, v, adjacency_matrix[u][v])

# Example usage with any source node
src = 1  # Replace with the desired source node index
distances, paths = g.dijkstra(src)
print(f"Shortest paths from source {src}:\n")
print("Vertex \tDistance from Source \tPath")
for node in range(len(distances)):
    path = paths[node] + [node]
    print(node, "\t", distances[node], "\t\t", path)