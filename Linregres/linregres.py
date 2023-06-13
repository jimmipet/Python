import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
# читаем файл с данными
data = pd.read_csv('cost_revenue_clean.csv')

# создаем график по данным  и показываем его
plt.scatter(data.production_budget_usd, data.worldwide_gross_usd)
plt.xlabel("Бюджет")
plt.ylabel("Кассовый сбор")
plt.show()

# создаем модель
model = LinearRegression()
# обучаем нашу модель
x = pd.DataFrame(data.production_budget_usd)
y = pd.DataFrame(data.worldwide_gross_usd)
# кормим нашей модели нужные данные
model.fit(x, y)

# Линейная регрессия
plt.scatter(data.production_budget_usd, data.worldwide_gross_usd)
plt.plot(x, model.predict(x), color='red')
plt.xlabel("Бюджет")
plt.ylabel("Кассовый сбор")
plt.show()


# точность моедли (коэфицент от 0 до 1)
print(model.score(x, y))
