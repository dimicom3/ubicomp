#!/usr/bin/env python3

import re
import os
from typing import NamedTuple

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_mqtt import Mqtt
from influxdb import InfluxDBClient
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

app.config['MQTT_BROKER_URL'] = MQTT_ADDRESS
app.config['MQTT_BROKER_PORT'] = 1883
app.config['MQTT_USERNAME'] = MQTT_USER
app.config['MQTT_PASSWORD'] = MQTT_PASSWORD
app.config['MQTT_KEEPALIVE'] = 5
app.config['MQTT_TLS'] = None
app.config['MQTT_CLEAN_SESSION'] = True

socketio = SocketIO(app)
mqtt = Mqtt(app)

@app.route('/')
def index():
    return render_template('index.html')

class SensorData(NamedTuple):
    location: str
    measurement: str
    value: float
class MicData(NamedTuple):
    location: str
    value: str

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print('Connected with result code ' + str(rc))
    mqtt.subscribe(MQTT_TOPIC)

@mqtt.on_message()
def handle_message(client, userdata, message):
    print(message.topic + ' ' + str(message.payload))
    sensor_data = _parse_mqtt_message(message.topic, message.payload.decode('utf-8'))
    if type(sensor_data) is SensorData:
        _send_sensor_data_to_influxdb(sensor_data)
        _cache_sensor_data(sensor_data)
    elif type(sensor_data) is MicData:
        socketio.emit('mic', {'voice': sensor_data.value})




def _parse_mqtt_message(topic, payload):
    match = re.match(MQTT_REGEX, topic)
    if match:
        location = match.group(1)
        measurement = match.group(2)
        if measurement == 'status' or measurement == 'command':
            return None
        if measurement == 'mic':
            return MicData(location, str(payload))
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

def _cache_sensor_data(sensor_data):
    location = sensor_data.location
    if location not in sensor_cache:
        sensor_cache[location] = {}

    sensor_cache[location][sensor_data.measurement] = sensor_data.value

    if 'temperature' in sensor_cache[location] and 'humidity' in sensor_cache[location]:
        temperature = sensor_cache[location]['temperature']
        humidity = sensor_cache[location]['humidity']
        _process_and_respond(location, temperature, humidity)
        sensor_cache[location].pop('temperature')
        sensor_cache[location].pop('humidity')

def _process_and_respond(location, temperature, humidity):
    input_data = np.array([[temperature, humidity]], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    status = "Bad"
    if output_data[0] < 0.5:
        response_msg = f'Alert: Temperature and humidity in {location} are bad'
        mqtt.publish(MQTT_RESPONSE_TOPIC, response_msg)
        status = "Bad"
    else:
        status = "Good"

    socketio.emit('data', {'temperature': temperature, 'humidity': humidity, 'status': status})

_init_influxdb_database()
socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

