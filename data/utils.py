"""
Утилиты для загрузки стандартных датасетов
"""

import numpy as np
import urllib.request
import gzip
import os
from typing import Tuple
from .dataset import ArrayDataset, Dataset


def load_mnist(data_path: str = './data') -> Tuple[Dataset, Dataset]:
    """
    Загружает датасет MNIST

    Returns:
        train_dataset, test_dataset
    """
    os.makedirs(data_path, exist_ok=True)

    # URLs для MNIST
    urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    def download_file(url, filename):
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        return filepath

    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int64)

    # Загрузка данных
    train_images = load_images(download_file(urls['train_images'], 'train-images-idx3-ubyte.gz'))
    train_labels = load_labels(download_file(urls['train_labels'], 'train-labels-idx1-ubyte.gz'))
    test_images = load_images(download_file(urls['test_images'], 't10k-images-idx3-ubyte.gz'))
    test_labels = load_labels(download_file(urls['test_labels'], 't10k-labels-idx1-ubyte.gz'))

    train_dataset = ArrayDataset(train_images, train_labels)
    test_dataset = ArrayDataset(test_images, test_labels)

    return train_dataset, test_dataset


def load_iris() -> Dataset:
    """
    Загружает датасет Iris
    """
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    # Нормализация
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return ArrayDataset(X, y)


def load_california_housing() -> Dataset:
    """
    Загружает датасет California Housing для регрессии
    """
    from sklearn.datasets import fetch_california_housing

    housing = fetch_california_housing()
    X = housing.data.astype(np.float32)
    y = housing.target.astype(np.float32).reshape(-1, 1)

    # Нормализация
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return ArrayDataset(X, y)


def create_train_val_split(
        dataset: Dataset,
        val_ratio: float = 0.2,
        shuffle: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Разделяет датасет на тренировочный и валидационный
    """
    splits = dataset.split([1 - val_ratio, val_ratio], shuffle=shuffle)
    return splits[0], splits[1]