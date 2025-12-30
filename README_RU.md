# ml-credit-classifier

## Описание проекта

Это ML-проект для классификации кредитных рисков с использованием scikit-learn и XGBoost. Модель предсказывает вероятность дефолта заемщика на основе его финансовых характеристик.

## Особенности

- **XGBoost классификатор** для высокоточного прогнозирования
- **Docker контейнеризация** для простого развертывания
- **Стандартизация признаков** для улучшения производительности модели
- **Модульная архитектура** с отдельными скриптами для обучения и предсказания

## Установка и использование

### 1. Клонирование репозитория

```bash
git clone https://github.com/Cyb3rFake/ml-credit-classifier.git
cd ml-credit-classifier
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Обучение модели

```bash
python train.py
```

Скрипт `train.py` выполняет следующие операции:

1. **Загрузка данных** из CSV файла (`data/credit_data.csv`)
2. **Обработка пропусков** - заполнение средними значениями
3. **Разделение данных** на обучающий (80%) и тестовый (20%) наборы
4. **Стандартизация признаков** с помощью StandardScaler
5. **Обучение модели** XGBoost с параметрами:
   - n_estimators=100 (100 деревьев)
   - max_depth=5 (максимальная глубина дерева)
   - learning_rate=0.1 (скорость обучения)
6. **Оценка** на тестовом наборе с метриками классификации
7. **Сохранение** модели и scaler в `models/` директорию

### 4. Предсказание

```python
from predict import CreditPredictor

predictor = CreditPredictor(
    'models/credit_classifier.pkl',
    'models/scaler.pkl'
)

sample = {
    'age': 35,
    'income': 50000,
    'credit_score': 720,
    'debt_ratio': 0.3,
    'num_accounts': 4
}

result = predictor.predict_single(sample)
print(result)  # {'risk': 'Low', 'probability': 0.85, 'raw_prediction': 0}
```

## Структура проекта

```
ml-credit-classifier/
├── main.py                 # Главный класс CreditClassifier
├── train.py               # Скрипт для обучения модели
├── predict.py             # Скрипт для предсказания
├── requirements.txt        # Зависимости проекта
├── docker-compose.yml      # Docker compose конфиг
├── Dockerfile             # Dockerfile для контейнеризации
├── .gitignore             # Git ignore файл
└── README.md              # Документация на английском
```

## Docker использование

```bash
# Построение образа
docker build -t ml-credit-classifier .

# Запуск контейнера
docker run -v $(pwd)/data:/app/data ml-credit-classifier python train.py
```

Или с docker-compose:

```bash
docker-compose up
```

## Объяснение кода

### main.py

Главный модуль содержит класс `CreditClassifier`:

```python
class CreditClassifier:
    def __init__(self, model_type='xgboost'):
        # Инициализация классификатора
        self.model = XGBClassifier(...) if model_type == 'xgboost'
    
    def transform(self, X_new):
        # Масштабирование новых данных
        return self.scaler.transform(X_new)
    
    def predict_proba(self, X_scaled):
        # Вероятности классов
        return self.model.predict_proba(X_scaled)[:, 1]
```

### train.py

Процесс обучения:

1. **load_and_prepare_data()** - загрузка и подготовка данных
2. **train_model()** - создание и обучение модели
3. **evaluate_model()** - оценка производительности
4. Сохранение модели и scaler

### predict.py

Предсказание для новых данных:

```python
predictor = CreditPredictor('models/credit_classifier.pkl')
result = predictor.predict_single({'age': 35, 'income': 50000})
```

## Метрики оценки

- **Accuracy** - доля правильных предсказаний
- **Precision** - доля верно предсказанных позитивных случаев
- **Recall** - доля найденных позитивных случаев
- **F1-score** - гармоническое среднее precision и recall
- **Confusion Matrix** - матрица ошибок

## Требования

- Python 3.8+
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- joblib >= 1.1.0

## Лицензия

MIT License

## Контакты

Вопросы и предложения: GitHub Issues
