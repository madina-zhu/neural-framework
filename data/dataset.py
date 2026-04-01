"""
Модуль для работы с датасетами и загрузчиками данных
Реализует классы Dataset и DataLoader с поддержкой batch, shuffle, map
"""

import numpy as np
from typing import List, Tuple, Any, Callable, Optional, Iterator
from abc import ABC, abstractmethod


class Dataset(ABC):
    """Абстрактный базовый класс для датасетов"""

    @abstractmethod
    def __len__(self) -> int:
        """Возвращает размер датасета"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает пару (X, y) по индексу"""
        pass

    def map(self, func: Callable) -> 'MappedDataset':
        """Применяет функцию к каждому элементу датасета"""
        return MappedDataset(self, func)

    def filter(self, func: Callable) -> 'FilteredDataset':
        """Фильтрует элементы датасета"""
        return FilteredDataset(self, func)

    def split(self, ratios: List[float], shuffle: bool = True) -> List['SubsetDataset']:
        """Разделяет датасет на несколько подмножеств"""
        indices = list(range(len(self)))
        if shuffle:
            np.random.shuffle(indices)

        splits = []
        start = 0
        for ratio in ratios:
            size = int(ratio * len(self))
            split_indices = indices[start:start + size]
            splits.append(SubsetDataset(self, split_indices))
            start += size

        # Добавляем остаток в последний сплит, если есть
        if start < len(self):
            splits[-1] = SubsetDataset(self, indices[start:])

        return splits


class ArrayDataset(Dataset):
    """Датасет из массивов NumPy"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        assert len(X) == len(y), "X и y должны иметь одинаковую длину"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]


class MappedDataset(Dataset):
    """Датасет с примененной функцией к каждому элементу"""

    def __init__(self, dataset: Dataset, func: Callable):
        self.dataset = dataset
        self.func = func

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self.dataset[idx]
        return self.func(X, y)


class FilteredDataset(Dataset):
    """Фильтрованный датасет"""

    def __init__(self, dataset: Dataset, func: Callable):
        self.dataset = dataset
        self.func = func
        self.indices = [i for i in range(len(dataset)) if func(dataset[i])]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]


class SubsetDataset(Dataset):
    """Подмножество датасета по индексам"""

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.dataset[self.indices[idx]]


class DataLoader:
    """
    Загрузчик данных с поддержкой:
    - батчирования (batch)
    - перемешивания (shuffle)
    - параллельной загрузки (num_workers)
    """

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 32,
            shuffle: bool = False,
            num_workers: int = 0,
            drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self._indices = None

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Итератор по батчам"""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            batch_X = []
            batch_y = []
            for idx in batch_indices:
                X, y = self.dataset[idx]
                batch_X.append(X)
                batch_y.append(y)

            yield np.array(batch_X), np.array(batch_y)

    def __len__(self) -> int:
        """Количество батчей в загрузчике"""
        n_batches = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            n_batches += 1
        return n_batches

    def map(self, func: Callable) -> 'DataLoader':
        """Применяет функцию к каждому батчу"""
        self.dataset = self.dataset.map(func)
        return self

    def to_device(self, device: str = 'cpu'):
        """Перемещает данные на устройство (для совместимости с GPU)"""
        # Здесь можно добавить логику для GPU, если потребуется
        return self