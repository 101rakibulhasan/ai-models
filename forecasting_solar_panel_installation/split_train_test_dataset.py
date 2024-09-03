import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset/filtered.csv')

x = dataset.drop(columns=['count'])

y = dataset['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

train_data = pd.concat([y_train, x_train], axis=1)
test_data = pd.concat([y_test, x_test], axis=1)

train_data.to_csv('dataset/train_data.csv', index=False)
test_data.to_csv('dataset/test_data.csv', index=False)
