import numpy as np
from typing import List, Tuple


class GraphKnn:
    """
    Класс для создания и анализа графов на основе k-ближайших соседей.

    Граф строится путем сортировки значений входного массива и соединения
    каждого элемента с его k-ближайшими соседями (по значению).
    """

    def __init__(self, ksi: np.ndarray, k: int = 5):
        """
        Инициализация графа k-ближайших соседей.

        Args:
            ksi: Одномерный массив значений для построения графа.
            k: Количество ближайших соседей для соединения с каждым узлом.
        """
        self.n: int = len(ksi)
        self.k: int = k
        self.G: List[List[int]] = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.G_list: List[List[int]] = [[] for _ in range(self.n)]

        ksi = np.nan_to_num(ksi, nan=ksi.mean())

        tmp: List[Tuple[float, int]] = sorted([(ksi[i], i) for i in range(self.n)])
        self.points: List[int] = []
        for i in range(self.n):
            tmp_add: List[Tuple[float, int]] = []
            for j in range(max(0, i - k), min(self.n, i + k + 1)):
                if i == j:
                    continue
                tmp_add.append((abs(tmp[i][0] - tmp[j][0]), j))
            tmp_add.sort()
            for j in range(min(self.n - 1, k)):
                idx1: int = tmp[i][1]
                idx2: int = tmp_add[j][1]
                self.G[idx1][idx2] = 1
                self.G[idx2][idx1] = 1
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.G[i][j]:
                    self.G_list[i].append(j)
                    self.G_list[j].append(i)


        self.points = [x[1] for x in tmp]

    def calc_metric(self) -> int:
        """
        Вычисляет метрику графа, подсчитывая количество треугольников между соседями.

        Треугольник образуется, когда три вершины взаимосвязаны друг с другом.
        Наличие треугольников является показателем кластеризации графа.

        Returns:
            Количество треугольников в графе.
        """
        res: int = 0
        for v1 in range(self.n):
            for v2 in range(v1 + 1, self.n):
                for v3 in range(v2 + 1, self.n):
                    if self.G[self.points[v1]][self.points[v2]] and self.G[self.points[v1]][self.points[v3]] and self.G[self.points[v2]][self.points[v3]]:
                        res += 1
        return res

    def calc_connected_components(self) -> int:
        """
        Вычисляет количество компонент связности в графе.

        Компонента связности - это максимальное множество вершин,
        между которыми существует путь. Используется алгоритм поиска в глубину (DFS).

        Returns:
            Количество компонент связности в графе.
        """
        visited: List[bool] = [False] * self.n
        components: int = 0

        def dfs(v: int) -> None:
            visited[v] = True
            for u in self.G_list[v]:
                if not visited[u]:
                    dfs(u)

        for v in range(self.n):
            if not visited[v]:
                dfs(v)
                components += 1

        return components
