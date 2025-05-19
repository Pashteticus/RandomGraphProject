import numpy as np 

class GraphKnn:
    def __init__(self, ksi: np.ndarray, k: int=5):
        self.n = len(ksi)
        self.k = k 
        self.G = [[0 for _ in range(self.n)] for _ in range(self.n)]
        tmp = sorted([[ksi[i], i] for i in range(self.n)])
        for i in range(self.n):
            for j in range(i+1, min(self.n, i+k+1)):
                self.G[tmp[i][1]][tmp[j][1]] = 1
                self.G[tmp[j][1]][tmp[i][1]] = 1

    def calc_metric(self):
        res = 0
        for v1 in range(self.n):
            for v2 in range(v1+1, min(self.n, v1+1+self.k)):
                for v3 in range(v2+1, min(self.n, v2+1+self.k)):
                    if self.G[v1][v2] and self.G[v1][v3] and self.G[v2][v3]:
                        res += 1
        return res 