# RandomGraphProject

Исследовательский проект по анализу случайных графов с использованием статистических методов и машинного обучения

## 📊 Описание проекта

Данный проект представляет собой комплексное исследование случайных графов, построенных на основе различных вероятностных распределений и последующий их анализ

### Ключевые возможности

- **Генерация случайных графов** на основе различных вероятностных распределений:
  - Нормальное и распределение Стьюдента (Part 1)
  - Распределения Парето и Гамма (Part 2)
  
- **Два типа графов**:
  - **KNN-графы**: построенные по принципу k-ближайших соседей
  - **Distance-графы**: основанные на пороговых расстояниях
  
- **Метрики для анализа графов**:
  - Количество треугольников и минимальное кликовое покрытие (для KNN)
  - Количество компонент связности и хроматическое число (для Distance)
  
- **Статистическое тестирование** гипотез с ROC-анализом
- **Машинное обучение** для классификации типов графов
- **Моделирование Монте-Карло** для оценки статистических свойств

### Структура проекта

```
RandomGraphProject/
├── src/                          # Исходный код
│   ├── data.py                   # Генератор случайных данных
│   ├── monte_carlo.py            # Симуляции Монте-Карло
│   └── graph/                    # Модули для работы с графами
│       ├── knn.py               # KNN-графы
│       └── dist.py              # Distance-графы
├── notebooks/                    # Jupyter ноутбуки с экспериментами
│   ├── Part_1_experiments.ipynb  # Эксперименты: Normal vs Student-t
│   ├── Part_1_classification.ipynb # Классификация: Normal vs Student-t  
│   ├── Part_2_experiments.ipynb  # Эксперименты: Pareto vs Gamma
│   └── Part_2_classification.ipynb # Классификация: Pareto vs Gamma
├── requirements.txt              # Зависимости для pip
├── pyproject.toml               # Конфигурация проекта
└── README.md                    # Документация проекта
```

## 🚀 Установка и запуск

### Требования

- Python 3.8+
- uv (рекомендуемый менеджер пакетов)

1. **Установите uv** (если ещё не установлен):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2.**Клонируйте репозиторий**:

```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd RandomGraphProject
```

3. **Создайте виртуальное окружение и установите зависимости**:

```bash
uv venv venv -p 3.13
source .venv/bin/activate  # macOS/Linux
# или .venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

## 📚 Использование

1. **Базовое использование генератора**:

```python
from src.data import Generator

# Создание генератора
gen = Generator(v=3, alpha=1.0, size=100)

# Генерация данных
normal_data = gen.get_f()      # Нормальное распределение
student_data = gen.get_h()     # Распределение Стьюдента
```

2. **Создание и анализ графов**:

```python
from src.graph.knn import GraphKnn
from src.graph.dist import GraphDist

# KNN-граф
knn_graph = GraphKnn(normal_data, k=5)
triangles = knn_graph.calc_metric()

# Distance-граф  
dist_graph = GraphDist(normal_data, d=0.75)
chromatic_num = dist_graph.calc_metric()
```

3. **Симуляции Монте-Карло**:

```python
from src.monte_carlo import MonteCarlo

mc = MonteCarlo(generator=gen, num_simulations=1000)
results = mc.run_simulation('knn', {'k': 5}, lambda g: g.calc_metric())
```
