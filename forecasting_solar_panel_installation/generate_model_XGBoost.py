import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv("dataset/single_city.csv")

data['count'].plot()

train_data = data.iloc[:int(.99 * len(data)), :]
test_data = data.iloc[int(.99 * len(data)):, :]

feature ='installation_date'
target = "count"

model = xgb.XGBRegressor()
model.fit(train_data[feature], train_data[target])

predictions = model.predict(test_data[feature])
print("Model Predicitons")
print(predictions)

print('Actual Values: ')
print(test_data[target])

accuracy = model.score(test_data[feature], test_data[target])
print('Accuracy:')
print(accuracy)