import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


def get_current_file_directory():
    return os.path.dirname(os.path.abspath(__file__))


model_filename = 'classification_model.tflite'


current_dir = get_current_file_directory()

model_path = os.path.join(current_dir, model_filename)

if not os.path.exists(model_path):
    print(f"Model file does not exist at {model_path}")
    exit(1)

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Model loaded successfully.")
except ValueError as e:
    print(f"Error loading model: {e}")
    exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)


input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)

scaler = StandardScaler()

dummy_data = np.array([[0, 0], [45, 100]]) 
scaler.fit(dummy_data)

input_temperature = 24.0 
input_humidity = 44.0    

input_data = np.array([[input_temperature, input_humidity]])
input_data_normalized = scaler.transform(input_data)
print("Normalized input data:", input_data_normalized)


if input_data_normalized.shape != tuple(input_shape):
    print(f"Input data shape {input_data_normalized.shape} does not match expected shape {input_shape}")
    exit(1)

interpreter.set_tensor(input_details[0]['index'], input_data_normalized.astype(np.float32))

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output data:", output_data)

predicted_class = (output_data > 0.5).astype(int)
print("Predicted class:", predicted_class)
