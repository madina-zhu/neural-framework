"""
Тесты для модуля данных
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ArrayDataset, DataLoader
from data.transforms import Compose, Normalize, Flatten


def test_array_dataset():
    """Тест ArrayDataset"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    dataset = ArrayDataset(X, y)

    assert len(dataset) == 100
    X_sample, y_sample = dataset[0]
    assert X_sample.shape == (10,)
    assert y_sample.shape == ()
    assert np.array_equal(X_sample, X[0])
    assert y_sample == y[0]


def test_dataloader_batch():
    """Тест батчирования"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    dataset = ArrayDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    batches = list(loader)
    assert len(batches) == 4  # 100 / 32 = 3.125 -> 4 батча

    X_batch, y_batch = batches[0]
    assert X_batch.shape == (32, 10)
    assert y_batch.shape == (32,)


def test_dataloader_shuffle():
    """Тест перемешивания"""
    X = np.arange(100).reshape(-1, 1)
    y = np.arange(100)

    dataset = ArrayDataset(X, y)
    loader1 = DataLoader(dataset, batch_size=10, shuffle=False)
    loader2 = DataLoader(dataset, batch_size=10, shuffle=True)

    batches1 = list(loader1)
    batches2 = list(loader2)

    # Проверяем, что порядок разный
    first_batch1 = batches1[0][0].flatten()
    first_batch2 = batches2[0][0].flatten()

    assert not np.array_equal(first_batch1, first_batch2)


def test_dataloader_drop_last():
    """Тест drop_last"""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)

    dataset = ArrayDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, drop_last=True)

    batches = list(loader)
    assert len(batches) == 3  # 100 // 32 = 3

    X_batch, _ = batches[-1]
    assert X_batch.shape[0] == 32


def test_transforms_compose():
    """Тест композиции трансформаций"""
    X = np.random.randn(10, 28, 28)
    y = np.random.randint(0, 10, 10)

    transform = Compose([
        Flatten(),
        Normalize()
    ])

    X_transformed, y_transformed = transform(X[0], y[0])

    assert X_transformed.shape == (784,)
    assert np.abs(np.mean(X_transformed)) < 1e-5


if __name__ == "__main__":
    test_array_dataset()
    test_dataloader_batch()
    test_dataloader_shuffle()
    test_dataloader_drop_last()
    test_transforms_compose()
    print("✅ All tests passed!")