import numpy as np
from typing import Callable, List, Dict, Any
from data import Generator
from graph.knn import GraphKnn
from graph.dist import GraphDist


class MonteCarlo:
    """
    Класс для проведения симуляций Монте-Карло для анализа графов.
    """

    def __init__(self, generator: Generator, num_simulations: int = 1000):
        """
        Инициализация симулятора Монте-Карло.

        :param generator: Экземпляр класса Generator для генерации данных.
        :param num_simulations: Количество симуляций для проведения.
        """
        if not isinstance(generator, Generator):
            raise TypeError(
                "Параметр 'generator' должен быть экземпляром класса Generator."
            )
        if not isinstance(num_simulations, int) or num_simulations <= 0:
            raise ValueError(
                "Параметр 'num_simulations' должен быть положительным целым числом."
            )

        self.generator: Generator = generator
        self.num_simulations: int = num_simulations

    def run_simulation(
        self,
        graph_type: str,
        graph_params: Dict[str, Any],
        metric_fn: Callable[..., float],
    ) -> List[float]:
        """
        Запускает симуляцию для заданного типа графа и метрики.

        :param graph_type: Тип графа ('knn' или 'dist').
        :param graph_params: Словарь параметров для конструктора графа.
        :param metric_fn: Функция для вычисления метрики графа.
                          Принимает экземпляр графа в качестве аргумента.
        :return: Список значений метрики, полученных в ходе симуляций.
        :raises ValueError: Если указан неподдерживаемый тип графа.
        """
        if graph_type not in ["knn", "dist"]:
            raise ValueError(
                f"Неподдерживаемый тип графа: {graph_type}. Допустимые значения: 'knn', 'dist'."
            )
        if not callable(metric_fn):
            raise TypeError("Параметр 'metric_fn' должен быть вызываемой функцией.")

        results: List[float] = []
        for _ in range(self.num_simulations):
            ksi_f: np.ndarray = self.generator.get_f()
            current_ksi: np.ndarray = ksi_f

            graph: Any
            if graph_type == "knn":
                graph = GraphKnn(ksi=current_ksi, **graph_params)
            elif graph_type == "dist":
                graph = GraphDist(ksi=current_ksi, **graph_params)

            metric_value: float = metric_fn(graph)
            results.append(metric_value)

        return results


if __name__ == "__main__":
    data_generator = Generator(v=5, alpha=0.5, size=50, seed=123)
    mc_simulator = MonteCarlo(generator=data_generator, num_simulations=100)
    knn_params: Dict[str, Any] = {"k": 5}

    def knn_metric_fn(g: GraphKnn) -> float:
        return g.calc_metric()

    try:
        knn_results: List[float] = mc_simulator.run_simulation(
            graph_type="knn", graph_params=knn_params, metric_fn=knn_metric_fn
        )
        print(
            f"Результаты симуляции для GraphKnn (среднее): {np.mean(knn_results):.4f}"
        )
        print(
            f"Результаты симуляции для GraphKnn (стандартное отклонение): {np.std(knn_results):.4f}"
        )
    except (ValueError, TypeError) as e:
        print(f"Ошибка при симуляции GraphKnn: {e}")

    dist_params: Dict[str, Any] = {"d": 0.75}

    def dist_metric_fn(g: GraphDist) -> float:
        return g.calc_metric(strategies=["largest_first", "random_sequential"])

    try:
        dist_results: List[float] = mc_simulator.run_simulation(
            graph_type="dist", graph_params=dist_params, metric_fn=dist_metric_fn
        )
        print(
            f"Результаты симуляции для GraphDist (среднее): {np.mean(dist_results):.4f}"
        )
        print(
            f"Результаты симуляции для GraphDist (стандартное отклонение): {np.std(dist_results):.4f}"
        )
    except (ValueError, TypeError) as e:
        print(f"Ошибка при симуляции GraphDist: {e}")

    try:
        error_results = mc_simulator.run_simulation(
            graph_type="unknown", graph_params={}, metric_fn=lambda g: 0
        )
    except ValueError as e:
        print(f"Ожидаемая ошибка: {e}")
