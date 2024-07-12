import pandas as pd
import tensorflow as tf

# Import Dataset
dataset = pd.read_csv('dataset/test_data.csv')

x_test = dataset.drop(columns=['diagnosis(1=m, 0=b)'])

y_test = dataset['diagnosis(1=m, 0=b)']

model = tf.keras.models.load_model("cancerous_tumor_detection.keras")

model.evaluate(x_test, y_test)