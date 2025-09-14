import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # импортируем логистическую регрессию
from sklearn.metrics import precision_score, recall_score # импортируем метрики

data = pd.read_csv('data/scoring.csv')

print(data.head())

X = data.drop(columns=['default']).values # матрица признаков
y = data['default'].values # целевое значение - которое будем предсказывать

# деление на выборку обучения и выборку валидации 80 / 20 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

model = LogisticRegression(class_weight='balanced', random_state=42) # создаем модель для обучения
model.fit(X_train, y_train) # тренируем

y_pred = model.predict(X_test) # предсказываем

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Отказано {y_pred.mean() * 100:.0f}%")
print(f"Точность {precision * 100:.0f}%")
print(f"Полнота {recall * 100:.0f}%")

joblib.dump(model, 'model.pkl') # сохраняем тренированную модель в бинарном формате в файл
