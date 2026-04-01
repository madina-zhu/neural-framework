"""
Basic Usage Example - Как использовать фреймворк
Этот пример показывает минимальный код для обучения нейросети
"""

import sys
import os
# Добавляем корневую папку проекта в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Импортируем компоненты фреймворка
try:
    # Ваши модули (data)
    from data.dataset import DataLoader, ArrayDataset
    from data.transforms import Compose, Normalize, Flatten

    # Модули Кати (core) - если они уже есть
    from core.layers import Sequential, Linear
    from core.activations import ReLU, Softmax
    from core.losses import CrossEntropyLoss
    from core.optimizers import Adam

    CORE_AVAILABLE = True
    print("✅ Все компоненты фреймворка доступны")

except ImportError as e:
    print(f"⚠️ Некоторые компоненты не найдены: {e}")
    print("Будет запущена упрощенная версия без core")
    CORE_AVAILABLE = False

print("=" * 60)
print("Basic Usage Example - Демонстрация работы фреймворка")
print("=" * 60)

# 1. СОЗДАНИЕ ДАННЫХ
print("\n1. Создаем синтетические данные...")
X = np.random.randn(1000, 20)  # 1000 образцов, 20 признаков
y = np.random.randint(0, 2, 1000)  # 2 класса (0 или 1)
print(f"   Данные: X shape={X.shape}, y shape={y.shape}")

# 2. СОЗДАНИЕ DATASET И DATALOADER
print("\n2. Создаем Dataset и DataLoader...")
dataset = ArrayDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"   Датасет: {len(dataset)} образцов")
print(f"   DataLoader: {len(loader)} батчей")
print(f"   Первый батч: X shape={next(iter(loader))[0].shape}")

# 3. ПРОВЕРКА ТРАНСФОРМАЦИЙ
print("\n3. Проверяем трансформации...")
transform = Compose([
    Flatten(),
    Normalize()
])
X_sample, y_sample = dataset[0]
X_transformed, _ = transform(X_sample, y_sample)
print(f"   Исходная форма: {X_sample.shape}")
print(f"   После трансформаций: {X_transformed.shape}")

# 4. ЕСЛИ CORE ДОСТУПЕН - СОЗДАЕМ МОДЕЛЬ И ОБУЧАЕМ
if CORE_AVAILABLE:
    print("\n4. Строим нейросеть...")
    model = Sequential([
        Linear(20, 64),    # Входной слой: 20 -> 64 нейрона
        ReLU(),             # Функция активации ReLU
        Linear(64, 32),     # Скрытый слой: 64 -> 32 нейрона
        ReLU(),             # ReLU
        Linear(32, 2),      # Выходной слой: 32 -> 2 класса
        Softmax()           # Softmax для вероятностей
    ])
    print("   ✅ Модель создана")

    # Подсчет параметров
    total_params = sum(p.size for p in model.parameters())
    print(f"   Всего параметров: {total_params:,}")

    # 5. НАСТРОЙКА ОБУЧЕНИЯ
    print("\n5. Настраиваем обучение...")
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    print("   ✅ Оптимизатор и функция потерь настроены")

    # 6. ОБУЧЕНИЕ
    print("\n6. Запускаем обучение (10 эпох)...")
    try:
        history = model.fit(
            loader,      # Данные для обучения
            loss_fn,     # Функция потерь
            optimizer,   # Оптимизатор
            epochs=10,   # Количество эпох
            verbose=True # Показывать прогресс
        )
        print("   ✅ Обучение завершено")

        # 7. ПРЕДСКАЗАНИЕ
        print("\n7. Делаем предсказания...")
        predictions = model.predict(X[:5])
        print("   Предсказания (вероятности классов):")
        for i, pred in enumerate(predictions):
            class_pred = np.argmax(pred)
            print(f"   Образец {i+1}: класс0={pred[0]:.3f}, класс1={pred[1]:.3f} -> {class_pred}")

    except Exception as e:
        print(f"   ❌ Ошибка при обучении: {e}")
        print("   Возможно, метод fit еще не реализован в core")
else:
    print("\n⚠️ Core модули не найдены. Демонстрация только вашей части:")
    print("   ✅ Dataset и DataLoader работают")
    print("   ✅ Трансформации работают")
    print("   ✅ Загрузка данных работает")
    print("\n📌 Когда Катя добавит core модули, вы сможете:")
    print("   - Создавать нейросети")
    print("   - Обучать модели")
    print("   - Делать предсказания")

print("\n" + "=" * 60)
print("✅ Basic usage example завершен!")
print("\n🎯 Что дальше?")
print("1. Запустите: python examples/iris_demo.py")
print("2. Запустите: streamlit run web_demo/app.py")
print("3. Добавьте core модули от Кати")
print("=" * 60)