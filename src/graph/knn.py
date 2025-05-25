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

        valid_indices: np.ndarray = np.isfinite(ksi)
        ksi_clean: np.ndarray
        if not np.all(valid_indices):
            valid_values: np.ndarray = ksi[valid_indices]
            if len(valid_values) > 0:
                median_val: float = np.median(valid_values)
                ksi_clean = np.where(valid_indices, ksi, median_val)
            else:
                ksi_clean = np.zeros_like(ksi)
        else:
            ksi_clean = ksi

        tmp: List[Tuple[float, int]] = sorted(
            [(ksi_clean[i], i) for i in range(self.n)]
        )

        for i in range(self.n):
            for j in range(i + 1, min(self.n, i + k + 1)):
                idx1: int = tmp[i][1]
                idx2: int = tmp[j][1]
                self.G[idx1][idx2] = 1
                self.G[idx2][idx1] = 1

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
            for v2_offset in range(1, self.k + 1):
                v2: int = v1 + v2_offset
                if v2 >= self.n:
                    break

                for v3_offset in range(v2_offset + 1, self.k + 1):
                    v3: int = v1 + v3_offset
                    if v3 >= self.n:
                        break
                    if self.G[v1][v2] and self.G[v1][v3] and self.G[v2][v3]:
                        res += 1
        return res
