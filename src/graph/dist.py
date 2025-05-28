import numpy as np
import networkx as nx
from typing import List, Tuple, Any, Dict


class GraphDist:
    """
    Класс для создания и анализа графов на основе пороговых расстояний.

    Граф строится путем соединения вершин, расстояние между которыми
    превышает заданное пороговое значение d. Использует библиотеку NetworkX
    для представления и анализа графа.
    """

    def __init__(self, ksi: np.ndarray, d: float = 0.5):
        """
        Инициализация графа на основе пороговых расстояний.

        Args:
            ksi: Одномерный массив значений для построения графа.
            d: Пороговое расстояние для определения связей между вершинами.
        """
        self.n: int = len(ksi)
        self.d: float = d
        self.G = nx.Graph()

        ksi = np.nan_to_num(ksi, nan=ksi.mean())

        tmp: List[Tuple[float, int]] = sorted([(ksi[i], i) for i in range(self.n)])
        for i in range(self.n):
            self.G.add_node(i)
        dop: List[List[int]] = [[1 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if tmp[j][0] - tmp[i][0] > self.d:
                    break
                dop[tmp[i][1]][tmp[j][1]] = 0
                dop[tmp[j][1]][tmp[i][1]] = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if dop[i][j]:
                    self.G.add_edge(i, j)

    def calc_metric(self, strategies: List[str] = None) -> float:
        """
        Вычисляет метрику графа, используя жадные алгоритмы раскраски графа.

        Метрика определяется как среднее количество цветов, необходимых для
        раскраски графа различными стратегиями.

        Args:
            strategies: Список стратегий раскраски графа для использования.
                       Если None, используется только стратегия 'largest_first'.

        Returns:
            Среднее количество цветов, необходимых для раскраски графа.

        Raises:
            TypeError: Если параметр strategies не является списком строк.
        """
        if strategies is None:
            strategies = ["largest_first"]

        if not isinstance(strategies, list) or not all(
            isinstance(s, str) for s in strategies
        ):
            raise TypeError("Параметр 'strategies' должен быть списком строк.")

        results: List[int] = []
        for strategy in strategies:
            try:
                coloring: Dict[Any, int] = nx.greedy_color(self.G, strategy=strategy)
                if not coloring:
                    num_colors = 0 if self.n == 0 else 1
                else:
                    num_colors = max(coloring.values()) + 1
                results.append(num_colors)
            except nx.NetworkXError as e:
                print(f"Ошибка при раскраске графа со стратегией '{strategy}': {e}")
                results.append(0)

        if not results:
            return 0.0

        return float(np.mean(results))

    def calc_chromatic_number(self, strategy: str = "largest_first") -> int:
        """
        Вычисляет хроматическое число графа.

        Хроматическое число - это минимальное количество цветов, необходимых
        для раскраски вершин графа таким образом, чтобы никакие две смежные
        вершины не имели одинакового цвета.

        Args:
            strategy: Стратегия жадного алгоритма раскраски.
                     Доступные стратегии: 'largest_first', 'random_sequential',
                     'smallest_last', 'independent_set', 'connected_sequential_bfs',
                     'connected_sequential_dfs', 'saturation_largest_first'.

        Returns:
            Хроматическое число графа (приближённое значение).

        Raises:
            TypeError: Если параметр strategy не является строкой.
        """
        if not isinstance(strategy, str):
            raise TypeError("Параметр 'strategy' должен быть строкой.")

        try:
            coloring: Dict[Any, int] = nx.greedy_color(self.G, strategy=strategy)
            if not coloring:
                return 0 if self.n == 0 else 1
            return max(coloring.values()) + 1
        except nx.NetworkXError as e:
            print(f"Ошибка при раскраске графа со стратегией '{strategy}': {e}")
            return 0
