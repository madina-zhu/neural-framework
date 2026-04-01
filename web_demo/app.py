"""
No-Code веб-интерфейс для обучения нейросетей
Стиль TensorFlow Playground
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DataLoader, ArrayDataset
from data.utils import load_mnist, load_iris, load_california_housing
from data.transforms import Compose, Normalize, Flatten
from core.layers import Sequential, Linear
from core.activations import ReLU, Sigmoid, Tanh, Softmax
from core.losses import MSELoss, CrossEntropyLoss
from core.optimizers import SGD, MomentumSGD, Adam
from core.optimizers import GradientClipping

# Настройка страницы
st.set_page_config(
    page_title="Neural Framework Playground",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Framework Playground")
st.markdown("No-code нейросетевой фреймворк — обучайте нейросети без программирования!")

# Боковая панель с настройками
with st.sidebar:
    st.header("⚙️ Configuration")

    # Выбор датасета
    dataset_name = st.selectbox(
        "Dataset",
        ["Iris (Classification)", "MNIST (Classification)", "California Housing (Regression)"]
    )

    # Настройка модели
    st.subheader("Model Architecture")

    hidden_layers = st.number_input("Number of hidden layers", min_value=1, max_value=5, value=2)

    layers_config = []
    for i in range(hidden_layers):
        col1, col2 = st.columns(2)
        with col1:
            neurons = st.number_input(f"Layer {i + 1} neurons", min_value=4, max_value=512, value=64)
        with col2:
            activation = st.selectbox(
                f"Activation {i + 1}",
                ["ReLU", "Sigmoid", "Tanh"],
                key=f"act_{i}"
            )
        layers_config.append((neurons, activation))

    # Оптимизатор
    st.subheader("Optimizer")
    optimizer_name = st.selectbox("Optimizer", ["SGD", "Momentum SGD", "Adam"])

    lr = st.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.01, format="%.4f")

    if optimizer_name == "Momentum SGD":
        momentum = st.slider("Momentum", 0.0, 0.99, 0.9)

    use_clipping = st.checkbox("Use Gradient Clipping")
    if use_clipping:
        clip_norm = st.number_input("Clip norm", min_value=0.1, max_value=10.0, value=1.0)

    # Параметры обучения
    st.subheader("Training")
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128, 256])
    epochs = st.slider("Epochs", 10, 200, 50)
    val_split = st.slider("Validation split", 0.1, 0.3, 0.2)

    train_button = st.button("🚀 Train Model", type="primary")

# Основная область
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Preview")

    # Загрузка датасета
    if dataset_name == "Iris (Classification)":
        dataset = load_iris()
        input_size = 4
        output_size = 3
        task = "classification"
        st.write(f"**Iris Dataset**: {len(dataset)} samples, 4 features, 3 classes")

        # Показываем примеры
        X_sample, y_sample = dataset[:5]
        st.dataframe(np.hstack([X_sample, y_sample.reshape(-1, 1)]),
                     columns=[f"Feature {i}" for i in range(4)] + ["Class"])

    elif dataset_name == "MNIST (Classification)":
        dataset, _ = load_mnist()
        input_size = 784
        output_size = 10
        task = "classification"
        st.write(f"**MNIST Dataset**: {len(dataset)} samples, 28x28 images, 10 digits")

        # Показываем примеры изображений
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            X, y = dataset[i]
            axes[i].imshow(X.reshape(28, 28), cmap='gray')
            axes[i].set_title(f"Digit: {y}")
            axes[i].axis('off')
        st.pyplot(fig)
        plt.close()

    else:  # California Housing
        dataset = load_california_housing()
        input_size = 8
        output_size = 1
        task = "regression"
        st.write(f"**California Housing Dataset**: {len(dataset)} samples, 8 features, 1 target")

with col2:
    st.subheader("🎯 Model Summary")

    # Создаем модель
    layers = []

    # Входной слой
    layers.append(Linear(input_size, layers_config[0][0]))

    # Скрытые слои
    for i, (neurons, activation) in enumerate(layers_config):
        if activation == "ReLU":
            layers.append(ReLU())
        elif activation == "Sigmoid":
            layers.append(Sigmoid())
        elif activation == "Tanh":
            layers.append(Tanh())

        if i < len(layers_config) - 1:
            layers.append(Linear(layers_config[i][0], layers_config[i + 1][0]))

    # Выходной слой
    layers.append(Linear(layers_config[-1][0], output_size))

    if task == "classification" and output_size > 1:
        layers.append(Softmax())

    model = Sequential(layers)

    # Показываем архитектуру
    st.text(str(model))

    total_params = sum(p.size for p in model.parameters())
    st.info(f"**Total parameters**: {total_params:,}")

# Обучение
if train_button:
    st.header("📈 Training")

    # Разделение данных
    from data.utils import create_train_val_split

    train_dataset, val_dataset = create_train_val_split(dataset, val_ratio=val_split)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    if task == "classification":
        loss_fn = CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    # Optimizer
    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Momentum SGD":
        optimizer = MomentumSGD(model.parameters(), lr=lr, momentum=momentum)
    else:  # Adam
        optimizer = Adam(model.parameters(), lr=lr)

    if use_clipping:
        optimizer = GradientClipping(optimizer, max_norm=clip_norm)

    # Прогресс
    progress_bar = st.progress(0)
    loss_placeholder = st.empty()

    # Для визуализации
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Обучение
    for epoch in range(epochs):
        # Обучение
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Forward
            predictions = model.forward(X_batch)
            loss = loss_fn.forward(predictions, y_batch)

            # Backward
            model.zero_grad()
            grad = loss_fn.backward()
            model.backward(grad)

            # Update weights
            optimizer.step()

            epoch_loss += loss.item()

            if task == "classification":
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch
                correct += np.sum(pred_classes == true_classes)
                total += len(y_batch)

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if task == "classification":
            train_acc = correct / total
            train_accs.append(train_acc)

        # Валидация
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        for X_batch, y_batch in val_loader:
            predictions = model.forward(X_batch)
            loss = loss_fn.forward(predictions, y_batch)
            val_loss += loss.item()

            if task == "classification":
                pred_classes = np.argmax(predictions, axis=1)
                true_classes = np.argmax(y_batch, axis=1) if y_batch.ndim > 1 else y_batch
                val_correct += np.sum(pred_classes == true_classes)
                val_total += len(y_batch)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if task == "classification":
            val_acc = val_correct / val_total
            val_accs.append(val_acc)

        # Обновляем прогресс
        progress_bar.progress((epoch + 1) / epochs)
        loss_placeholder.info(f"Epoch {epoch + 1}/{epochs} - "
                              f"Train Loss: {avg_train_loss:.4f} - "
                              f"Val Loss: {avg_val_loss:.4f}" +
                              (f" - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
                               if task == "classification" else ""))

    # Визуализация результатов
    st.subheader("📉 Training Curves")

    fig, axes = plt.subplots(1, 2 if task == "classification" else 1, figsize=(12, 4))

    # Потери
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if task == "classification":
        # Точность
        axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
        axes[1].plot(val_accs, label='Val Accuracy', linewidth=2, linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Финальная оценка
    st.subheader("✅ Final Results")
    if task == "classification":
        st.success(f"**Final Validation Accuracy**: {val_accs[-1]:.4f}")
    else:
        st.success(f"**Final Validation Loss**: {val_losses[-1]:.4f}")

    # Сохраняем модель
    if st.button("💾 Save Model"):
        import pickle

        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        st.success("Model saved as 'trained_model.pkl'")

# Информация
st.sidebar.markdown("---")
st.sidebar.info(
    "**Neural Framework v1.0**\n\n"
    "Built with ❤️ using:\n"
    "- Custom autograd engine\n"
    "- NumPy for computations\n"
    "- Streamlit for UI\n\n"
    "Made by: Катя & Мадина"
)