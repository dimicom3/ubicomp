#!/usr/bin/env python3


import re
from typing import NamedTuple

import paho.mqtt.client as mqtt
from influxdb import InfluxDBClient
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from flask import Flask, render_template
from flask_socketio import SocketIO

INFLUXDB_ADDRESS = 'influxdb'
INFLUXDB_USER = 'root'
INFLUXDB_PASSWORD = 'root'
INFLUXDB_DATABASE = 'home_db'

MQTT_ADDRESS = 'mosquitto'
MQTT_USER = 'mqttuser'
MQTT_PASSWORD = 'mqttpassword'
MQTT_TOPIC = 'home/+/+'  # [bme280|mijia]/[temperature|humidity|battery|status]
MQTT_REGEX = 'home/([^/]+)/([^/]+)'
MQTT_CLIENT_ID = 'MQTTInfluxDBBridge'

MQTT_RESPONSE_TOPIC = 'home/dht22/command'

influxdb_client = InfluxDBClient(INFLUXDB_ADDRESS, 8086, INFLUXDB_USER, INFLUXDB_PASSWORD, None)


model_filename = 'classification_model.tflite'
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, model_filename)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sensor_cache = {}

# scaler = StandardScaler()
# dummy_data = np.array([[0, 0], [45, 100]]) 
# scaler.fit(dummy_data)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

class SensorData(NamedTuple):
    location: str
    measurement: str
    value: float


def on_connect(client, userdata, flags, rc):
    print('Connected with result code ' + str(rc))
    client.subscribe(MQTT_TOPIC)


def on_message(client, userdata, msg):
    print(msg.topic + ' ' + str(msg.payload))
    sensor_data = _parse_mqtt_message(msg.topic, msg.payload.decode('utf-8'))
    if sensor_data is not None:
        _send_sensor_data_to_influxdb(sensor_data)
        _cache_sensor_data(sensor_data, client)

def _parse_mqtt_message(topic, payload):
    match = re.match(MQTT_REGEX, topic)
    if match:
        location = match.group(1)
        measurement = match.group(2)
        if measurement == 'status' or measurement == 'command':
            return None
        return SensorData(location, measurement, float(payload))
    else:
        return None


def _send_sensor_data_to_influxdb(sensor_data):
    json_body = [
        {
            'measurement': sensor_data.measurement,
            'tags': {
                'location': sensor_data.location
            },
            'fields': {
                'value': sensor_data.value
            }
        }
    ]
    influxdb_client.write_points(json_body)


def _init_influxdb_database():
    databases = influxdb_client.get_list_database()
    if len(list(filter(lambda x: x['name'] == INFLUXDB_DATABASE, databases))) == 0:
        influxdb_client.create_database(INFLUXDB_DATABASE)
    influxdb_client.switch_database(INFLUXDB_DATABASE)

def _cache_sensor_data(sensor_data, client):
    location = sensor_data.location
    if location not in sensor_cache:
        sensor_cache[location] = {}

    sensor_cache[location][sensor_data.measurement] = sensor_data.value

    if 'temperature' in sensor_cache[location] and 'humidity' in sensor_cache[location]:
        temperature = sensor_cache[location]['temperature']
        humidity = sensor_cache[location]['humidity']
        _process_and_respond(location, temperature, humidity, client)
        sensor_cache[location].pop('temperature')
        sensor_cache[location].pop('humidity')

def _process_and_respond(location, temperature, humidity, client):
    input_data = np.array([[temperature, humidity]], dtype=np.float32)
    input_data_normalized = scaler.transform(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data_normalized)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    status = "-"
    if output_data[0] < 0.5:
        response_msg = f'Alert: Temperature and humidity in {location} are bad'
        client.publish(MQTT_RESPONSE_TOPIC, response_msg)
        status = "-"
    else:
        status = "+"

    socketio.emit('data', {'temperature': temperature, 'humidity': humidity, 'status': status})
def main():
    _init_influxdb_database()

    mqtt_client = mqtt.Client(MQTT_CLIENT_ID)
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    mqtt_client.connect(MQTT_ADDRESS, 1883)
    mqtt_client.loop_forever()
    socketio.run(app, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    print('MQTT to InfluxDB bridge')
    main()