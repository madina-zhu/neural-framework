"""
Вспомогательные функции для визуализации обучения
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def plot_training_history(
        history: Dict[str, List[float]],
        figsize: tuple = (12, 4)
):
    """
    Визуализирует историю обучения

    Args:
        history: словарь с метриками (train_loss, val_loss, train_acc, val_acc)
        figsize: размер фигуры
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Потери
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Точность
    if 'train_acc' in history:
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: List[str] = None
):
    """
    Визуализирует матрицу ошибок
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


def plot_mnist_samples(
        dataset,
        num_samples: int = 10,
        predictions: np.ndarray = None
):
    """
    Отображает примеры изображений MNIST
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        X, y = dataset[idx]
        img = X.reshape(28, 28)

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        title = f'True: {y}'
        if predictions is not None:
            title += f'\nPred: {predictions[idx]}'
        axes[i].set_title(title, fontsize=10)

    plt.tight_layout()
    plt.show()