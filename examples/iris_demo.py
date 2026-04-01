"""
Демонстрация работы фреймворка на датасете Iris
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.dataset import DataLoader, ArrayDataset
from data.transforms import Compose, Normalize, ToOneHot
from data.utils import load_iris, create_train_val_split
from examples.visualization import plot_training_history, plot_confusion_matrix


def iris_classification_demo():
    """
    Пример классификации на датасете Iris
    """
    print("=" * 60)
    print("Iris Classification Demo")
    print("=" * 60)

    # Загрузка данных
    print("\n1. Loading Iris dataset...")
    dataset = load_iris()
    print(f"   Dataset size: {len(dataset)}")

    # Разделение на train/val
    print("\n2. Splitting dataset...")
    train_dataset, val_dataset = create_train_val_split(dataset, val_ratio=0.2)
    print(f"   Train size: {len(train_dataset)}")
    print(f"   Val size: {len(val_dataset)}")

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Создание модели (используем фреймворк Кати)
    from core.layers import Sequential, Linear
    from core.activations import ReLU, Softmax
    from core.losses import CrossEntropyLoss
    from core.optimizers import Adam

    print("\n3. Creating model...")
    model = Sequential([
        Linear(4, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 3),
        Softmax()
    ])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    # Обучение
    print("\n4. Training model...")
    history = model.fit(
        train_loader,
        loss_fn,
        optimizer,
        epochs=100,
        verbose=True,
        val_loader=val_loader
    )

    # Визуализация
    print("\n5. Visualizing results...")
    plot_training_history(history)

    # Оценка на валидации
    print("\n6. Evaluating on validation set...")
    val_loss, val_acc = model.evaluate(val_loader, loss_fn)
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_acc:.4f}")

    # Получение предсказаний
    y_true = []
    y_pred = []
    for X_batch, y_batch in val_loader:
        predictions = model.predict(X_batch)
        y_true.extend(np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch)
        y_pred.extend(np.argmax(predictions, axis=1))

    plot_confusion_matrix(
        np.array(y_true),
        np.array(y_pred),
        classes=['Setosa', 'Versicolor', 'Virginica']
    )

    print("\n✅ Iris classification demo completed!")


def iris_regression_demo():
    """
    Пример регрессии на датасете Iris (предсказание длины лепестка)
    """
    print("\n" + "=" * 60)
    print("Iris Regression Demo (Predict petal length)")
    print("=" * 60)

    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data[:, [0, 1, 3]]  # Используем sepal length, sepal width, petal width
    y = iris.data[:, 2].reshape(-1, 1)  # Предсказываем petal length

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Нормализация
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    y_mean, y_std = y_train.mean(), y_train.std()

    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    train_dataset = ArrayDataset(X_train.astype(np.float32), y_train.astype(np.float32))
    val_dataset = ArrayDataset(X_val.astype(np.float32), y_val.astype(np.float32))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    from core.layers import Sequential, Linear
    from core.activations import ReLU
    from core.losses import MSELoss
    from core.optimizers import Adam

    model = Sequential([
        Linear(3, 64),
        ReLU(),
        Linear(64, 32),
        ReLU(),
        Linear(32, 1)
    ])

    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    history = model.fit(
        train_loader,
        loss_fn,
        optimizer,
        epochs=100,
        verbose=True,
        val_loader=val_loader
    )

    # Визуализация предсказаний
    plt.figure(figsize=(10, 6))
    y_val_pred = model.predict(X_val)
    y_val_pred = y_val_pred * y_std + y_mean
    y_val_original = y_val * y_std + y_mean

    plt.scatter(y_val_original, y_val_pred, alpha=0.6)
    plt.plot([y_val_original.min(), y_val_original.max()],
             [y_val_original.min(), y_val_original.max()], 'r--', lw=2)
    plt.xlabel('True Petal Length')
    plt.ylabel('Predicted Petal Length')
    plt.title('Regression: True vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    mse = np.mean((y_val_pred - y_val_original) ** 2)
    print(f"\nTest MSE: {mse:.4f}")


if __name__ == "__main__":
    iris_classification_demo()
    iris_regression_demo()