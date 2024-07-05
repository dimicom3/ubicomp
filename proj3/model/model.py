import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

num_samples = 10000
temperature = np.random.uniform(0, 45, num_samples)  
humidity = np.random.uniform(0, 100, num_samples)    

labels = np.where((temperature >= 20) & (temperature <= 25) & (humidity >= 30) & (humidity <= 50), 1, 0)

data = pd.DataFrame({
    'temperature': temperature,
    'humidity': humidity,
    'label': labels
})
print(data)
X = data[['temperature', 'humidity']]
y = data['label']

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('classification_model.tflite', 'wb') as f:
    f.write(tflite_model)
