import pandas as pd

data = pd.read_csv('E:\\GitHub\\ai-models\\satelite_installation_detection\\dataset\\data.csv')

x1 = data['city']
x2 = pd.to_datetime(data['installation_date'], dayfirst=True).dt.year

new_data = data.groupby([x1, x2]).size().reset_index(name='count')
new_data = new_data[(new_data['city'] != "-9999") & (new_data['installation_date'] != "-9999")]

new_data.to_csv('dataset/filtered.csv', index=False)