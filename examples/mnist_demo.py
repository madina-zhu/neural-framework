"""
Демонстрация работы фреймворка на датасете MNIST
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.dataset import DataLoader
from data.utils import load_mnist, create_train_val_split
from data.transforms import Compose, Normalize, Flatten
from examples.visualization import plot_training_history, plot_mnist_samples


def mnist_demo():
    """
    Пример классификации рукописных цифр MNIST
    """
    print("=" * 60)
    print("MNIST Digit Classification Demo")
    print("=" * 60)

    # Загрузка данных
    print("\n1. Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Test size: {len(test_dataset)}")

    # Показываем примеры
    print("\n2. Visualizing samples...")
    plot_mnist_samples(train_dataset, num_samples=5)

    # Разделение train на train/val
    print("\n3. Splitting into train/val...")
    train_dataset, val_dataset = create_train_val_split(train_dataset, val_ratio=0.1)
    print(f"   New train size: {len(train_dataset)}")
    print(f"   Validation size: {len(val_dataset)}")

    # Создание DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Создание модели
    from core.layers import Sequential, Linear
    from core.activations import ReLU, Softmax
    from core.losses import CrossEntropyLoss
    from core.optimizers import Adam
    from core.optimizers import GradientClipping

    print("\n4. Creating model...")
    model = Sequential([
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
        Softmax()
    ])

    loss_fn = CrossEntropyLoss()
    base_optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = GradientClipping(base_optimizer, max_norm=1.0)

    # Обучение
    print("\n5. Training model...")
    history = model.fit(
        train_loader,
        loss_fn,
        optimizer,
        epochs=20,
        verbose=True,
        val_loader=val_loader
    )

    # Визуализация
    print("\n6. Visualizing training history...")
    plot_training_history(history)

    # Оценка на тестовом наборе
    print("\n7. Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_loader, loss_fn)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")

    # Показываем предсказания на случайных примерах
    print("\n8. Showing predictions on random test samples...")
    predictions = []
    y_true = []
    for X_batch, y_batch in test_loader:
        pred = model.predict(X_batch)
        predictions.extend(np.argmax(pred, axis=1))
        y_true.extend(y_batch)

    plot_mnist_samples(test_dataset, num_samples=10, predictions=np.array(predictions))

    print("\n✅ MNIST demo completed!")


if __name__ == "__main__":
    mnist_demo()