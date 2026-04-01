"""
Функции для трансформации данных
"""

import numpy as np
from typing import Tuple, Callable


class Compose:
    """Композиция нескольких трансформаций"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y


class Normalize:
    """Нормализация данных в диапазон [0, 1]"""

    def __init__(self, mean: float = None, std: float = None):
        self.mean = mean
        self.std = std

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None:
            self.mean = np.mean(X)
        if self.std is None:
            self.std = np.std(X)

        X = (X - self.mean) / (self.std + 1e-8)
        return X, y


class Standardize:
    """Стандартизация данных (mean=0, std=1)"""

    def __init__(self, mean: float = None, std: float = None):
        self.mean = mean
        self.std = std

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mean is None:
            self.mean = np.mean(X)
        if self.std is None:
            self.std = np.std(X)

        X = (X - self.mean) / (self.std + 1e-8)
        return X, y


class ToOneHot:
    """Преобразование меток в one-hot encoding"""

    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_classes is None:
            self.num_classes = int(np.max(y) + 1)

        y_onehot = np.zeros((len(y) if y.ndim > 0 else 1, self.num_classes))
        if y.ndim == 0:
            y_onehot[0, int(y)] = 1
        else:
            for i, label in enumerate(y):
                y_onehot[i, int(label)] = 1

        return X, y_onehot


class Flatten:
    """Преобразование изображения в вектор"""

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X.ndim > 1:
            X = X.flatten()
        return X, y


class Reshape:
    """Изменение формы данных"""

    def __init__(self, shape: tuple):
        self.shape = shape

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = X.reshape(self.shape)
        return X, y


class Lambda:
    """Пользовательская трансформация"""

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.func(X, y)