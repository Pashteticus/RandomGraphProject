import numpy as np 
import networkx as nx 

class GraphDist:
    def __init__(self, ksi: np.ndarray, d: float=0.5):
        self.n = len(ksi)
        self.d = d
        self.G = nx.Graph()
        tmp = sorted([[ksi[i], i] for i in range(self.n)])
        for i in range(self.n):
            self.G.add_node(i)
        dop = [[1 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i+1, self.n):
                if tmp[j][0] - tmp[i][0] > self.d:
                    break 
                dop[tmp[i][1]][tmp[j][1]] = 0
                dop[tmp[j][1]][tmp[i][1]] = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if dop[i][j]:
                    self.G.add_edge(i, j)

    def calc_metric(self, strategies = ['largest_first']):
        res = [] 
        for strategy in strategies:
            res.append(len(nx.greedy_color(self.G, strategy=strategy)))
        return np.mean(res)