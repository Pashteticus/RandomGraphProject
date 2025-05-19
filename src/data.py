import numpy as np

class Generator:
    def __init__(self, v: int=3, alpha: int=1, size: int=25, seed: int=42):
        gen = np.random.default_rng(seed=seed)
        self.v = v
        self.alpha = alpha 
        self.size = size
        self.f = gen.normal
        self.h = gen.standard_t

    def get_f(self) -> np.ndarray:
        return self.f(0, self.alpha, self.size)
    
    def get_h(self) -> np.ndarray:
        return self.h(self.v, self.size)