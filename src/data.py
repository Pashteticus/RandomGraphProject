import numpy as np
from numpy.random import Generator as RNG


class Generator:
    """
    Генератор случайных последовательностей для построения и анализа графов.

    Генерирует последовательности из распределений:
    - нормальное распределение (f)
    - распределение Стьюдента (h)
    """

    def __init__(self, v: int = 3, alpha: float = 1.0, size: int = 25, seed: int = 42):
        """
        Инициализация генератора последовательностей.

        Args:
            v: Параметр степеней свободы для распределения Стьюдента.
            alpha: Стандартное отклонение для нормального распределения.
            size: Размер генерируемых последовательностей.
            seed: Начальное значение для генератора псевдослучайных чисел.
        """
        gen: RNG = np.random.default_rng(seed=seed)
        self.v: int = v
        self.alpha: float = alpha
        self.size: int = size
        self.f = lambda: gen.normal(0, self.alpha, self.size)
        self.h = lambda: gen.standard_t(self.v, self.size)

    def get_f(self) -> np.ndarray:
        """
        Генерирует выборку из нормального распределения N(0, alpha).

        Returns:
            Массив размера size из нормального распределения.
        """
        return self.f()

    def get_h(self) -> np.ndarray:
        """
        Генерирует выборку из распределения Стьюдента с v степенями свободы.

        Returns:
            Массив размера size из распределения Стьюдента.
        """
        return self.h()
