import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv("dataset/single_city.csv")

data['count'].plot()

data['installation_date'] = pd.to_datetime(data['installation_date'].astype(str) + '-01-01')
data.set_index('installation_date', inplace=True)

train_data = data['count'].iloc[:int(.99 * len(data))]
test_data = data['count'].iloc[int(.99 * len(data)):]

model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

predictions = model_fit.forecast(steps=len(test_data))
print("Model Predicitons")
print(predictions)

# Plotting the actual vs predicted values for visual comparison
plt.figure(figsize=(10,6))
plt.plot(test_data.index, test_data, label='Actual')
plt.plot(test_data.index, predictions, label='Predicted', color='red')
plt.legend()
plt.show()