import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import sys
from heapq import heappop, heappush

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
        nx.draw(self.G, pos=self.pos, with_labels=True)
        nx.draw_networkx_edge_labels(self.G, pos=self.pos, edge_labels= nx.get_edge_attributes(self.G,"weight"))
        plt.show()
        
        
        
G=graphvisualization()
G.addedge(2, 1, weight=10)
G.addedge(1, 3, weight=20)
G.addedge(1, 4, weight=22)
G.addedge(2, 5, weight=26)
G.addedge(3, 6, weight=35)
G.addedge(4, 7, weight=55)
G.addedge(5, 8, weight=43)
G.addedge(6, 8, weight=67)
G.addedge(7, 8, weight=150)
G.addedge(1, 9, weight=12)
G.addedge(2, 10, weight=27)
G.addedge(3, 11, weight=33)
G.addedge(9, 10, weight=190)
G.addedge(10, 7, weight=140)
G.addedge(11, 12, weight=15)
G.addedge(12, 13, weight=37)
G.addedge(12, 14, weight=26)
G.addedge(12, 15, weight=18)
G.addedge(13, 16, weight=38)
G.addedge(14, 17, weight=45)
G.addedge(15, 18, weight=56)
G.addedge(16, 19, weight=34)
G.addedge(17, 19, weight=43)
G.addedge(18, 19, weight=52)
G.addedge(11, 20, weight=102)
G.addedge(11, 21, weight=128)
G.addedge(11, 22, weight=49)
G.addedge(11, 23, weight=64)
G.addedge(20, 24, weight=78)
G.addedge(21, 25, weight=82)
G.addedge(22, 26, weight=53)
G.addedge(24, 27, weight=27)
G.addedge(25, 27, weight=16)
G.addedge(25, 27, weight=63)
G.addedge(25, 15, weight=97)
G.addedge(26, 4, weight=14)
G.addedge(9, 17, weight=62)
G.addedge(1, 12, weight=84)
G.addedge(3, 14, weight=92)

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
    distance=[random.randint(10,200) for x in range(m-1)]
    dist_edge=[distance[i] for i in range(m-1)]
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

G_new = nx.Graph()
weights = [
    4.5947261663286, 4.5947261663286, 4.5947261663286, 4.5947261663286, 4.5947261663286,
    4.498721227621484, 4.498721227621484, 3.5825179386823223, 3.5825179386823223, 3.5825179386823223,
    4.571333775713338, 5.947011070110701, 7.989414255469302, 4.85515873015873, 14.360279441117763,
    5.096537558685446, 3.467513611615245, 3.467513611615245, 9.806340718105423, 9.806340718105423,
    9.806340718105423, 9.806340718105423, 9.806340718105423, 4.494469408918078, 4.494469408918078,
    4.494469408918078, 9.557659932659933, 3.188811188811189, 6.941619585687382, 4.5992907801418434,
    3.794117647058824, 1.0294372294372296, 16.35897435897436, 4.432117274695985, 5.6689655172413795,
    10.541666666666666, 12.102240896358543, 4.5992907801418434
]
edges = [
    (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 8), (7, 8), (1, 9),
    (2, 10), (3, 11), (9, 10), (10, 7), (11, 12), (12, 13), (12, 14), (12, 15),
    (13, 16), (14, 17), (15, 18), (16, 19), (17, 19), (18, 19), (11, 20), (11, 21),
    (11, 22), (11, 23), (20, 24), (21, 25), (22, 26), (24, 27), (25, 27),
    (25, 15), (26, 4), (9, 17), (1, 12), (3, 14)
]

class Graph:
    def __init__(self):
        self.edges = edges
        self.weights = weights
        self.num_nodes = max(max(edge) for edge in self.edges)
        self.Q = np.zeros((self.num_nodes, self.num_nodes))

    def create_q_table(self):
        for edge, weight in zip(self.edges, self.weights):
            u, v = edge
            self.Q[u - 1, v - 1] = weight

    def print_q_table(self):
        q_table=self.Q
        return q_table

# Create the graph visualization object
G = Graph()

# Create the Q-table
G.create_q_table()

# Print the Q-table
Q_2=G.print_q_table()

df2=pd.DataFrame(Q_2)
#df2.to_csv('table2.csv', index=False)  

class QLearning:
    def __init__(self, edges, weights, learning_rate=0.1, discount_factor=0.9, num_episodes=100):
        self.edges = edges
        self.weights = weights
        self.num_nodes = max(max(edge) for edge in self.edges)
        self.Q = np.zeros((self.num_nodes, self.num_nodes))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes

    def update_q_table(self, u, v, reward, next_u):
        max_q = np.max(self.Q[next_u - 1])
        self.Q[u - 1, v - 1] = (1 - self.learning_rate) * self.Q[u - 1, v - 1] + self.learning_rate * (reward + self.discount_factor * max_q)

    def q_learning(self):
        for episode in range(self.num_episodes):
            for (u, v), reward in zip(self.edges, self.weights):
                next_u = v
                self.update_q_table(u, v, reward, next_u)

    def get_q_table(self):
        return self.Q
    
    def find_shortest_path(self, start_node, end_node):
        distances, _ = self.dijkstra(start_node)
        path = self._construct_path(end_node, distances)
        return path

    def _construct_path(self, end_node, distances):
        if distances[end_node - 1] == float("inf"):
            return None  # No path found
        path = [end_node]
        current_node = end_node
        while current_node != start_node:
            for prev_node in range(self.num_nodes):
                if self.Q[prev_node][current_node - 1] > 0 and distances[current_node - 1] - self.Q[prev_node][current_node - 1] == distances[prev_node]:
                    path.append(prev_node + 1)
                    current_node = prev_node + 1
                    break
        return path[::-1]  # Reverse the path

# Create the Q-learning object
q_learning = QLearning(edges, weights)

# Apply the Q-learning algorithm
q_learning.q_learning()

# Get the resulting Q-table
q_table = q_learning.get_q_table()
q_learning.Q = [[0,15.22523906,36.97918546,11.53834655,0,0,0,0,20.99584759,0,0,31.71345243,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [15.22523906,0,0,0,8.642150762,0,0,0,0,11.83611636,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [36.97918546,0,0,0,0,7.817833061,0,0,0,0,36.36129877,0,0,21.35127343,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [11.53834655,0,0,0,0,0,7.721830672,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12.61314789,0],
                [0,8.642150762,0,0,0,0,0,4.498601735,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,7.817833061,0,0,0,0,3.582422782,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,7.721830672,0,0,0,3.582422782,0,8.078353864,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,4.498601735,3.582422782,3.582422782,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [20.99584759,0,0,0,0,0,0,0,0,15.25410605,0,0,0,0,0,0,19.36451466,0,0,0,0,0,0,0,0,0,0],
                [0,11.83611636,0,0,0,0,8.078353864,0,15.25410605,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,36.36129877,0,0,0,0,0,0,0,0,33.9657945,0,0,0,0,0,0,0,11.57202035,22.22271133,23.10823325,3.18872649,0,0,0,0],
                [31.71345243,0,0,0,0,0,0,0,0,0,33.9657945,0,21.84695721,20.21797653,15.92372886,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,21.84695721,0,0,0,18.62894777,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,21.35127343,0,0,0,0,0,0,0,0,20.21797653,0,0,0,0,18.62894777,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,15.92372886,0,0,0,0,0,13.84980148,0,0,0,0,0,0,16.8891583,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,18.62894777,0,0,0,0,0,9.806080248,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,19.36451466,0,0,0,0,18.62894777,0,0,0,0,9.806080248,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,13.84980148,0,0,0,4.49435003,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9.806080248,9.806080248,4.49435003,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,11.57202035,0,0,0,0,0,0,0,0,0,0,0,0,7.867630671,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,22.22271133,0,0,0,0,0,0,0,0,0,0,0,0,0,19.77239668,0,0],
                [0,0,0,0,0,0,0,0,0,0,23.10823325,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15.12464121,0],
                [0,0,0,0,0,0,0,0,0,0,3.18872649,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7.867630671,0,0,0,0,0,0,1.029409886],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,16.8891583,0,0,0,0,0,19.77239668,0,0,0,0,0,16.35853984],
                [0,0,0,12.61314789,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1.029409886,16.35853984,0,0]
           ]
q_frame=pd.DataFrame(q_table)

#q_frame.to_csv('table5.csv', index=False)  #

def shortestPath(q_table,start_node,end_node):
    path=[start_node]
    next_node=np.argmax(q_frame.loc[start_node,])
    path.append(next_node)
    print(path)
    while next_node!=end_node:
        next_node=np.argmax(q_frame.loc[next_node,])
        path.append(next_node)
        print(path)
    return path


# Find and print the shortest path
start_node = 0
end_node = 27
current_node = start_node
optimal_path = [current_node]

while current_node != end_node:
    next_node = np.argmax(q_learning.Q[current_node])
    optimal_path.append(next_node)
    current_node = next_node

print("Optimal Path:", optimal_path)
total_cost_q_learning = 0
total_reward_q_learning = 0

for i in range(len(optimal_path) - 1):
    u, v = optimal_path[i], optimal_path[i + 1]
    total_cost_q_learning += c[u]
    # Using c values to get the cost
    total_reward_q_learning += q_table[u, v]  
    #print(total_reward_q_learning)# Using Q-values to get the reward

# Print the total cost and total reward of the optimal path using Q-learning
print("Total cost of optimal path using Q-learning:", total_cost_q_learning)
print("Total reward of optimal path using Q-learning:", total_reward_q_learning)
