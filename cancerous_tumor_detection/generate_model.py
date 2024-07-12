import pandas as pd
import tensorflow as tf

# Import Dataset
dataset = pd.read_csv('dataset/train_data.csv')

x_train = dataset.drop(columns=['diagnosis(1=m, 0=b)'])

y_train = dataset['diagnosis(1=m, 0=b)']

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)

model.save('model/cancerous_tumor_detection.keras')
