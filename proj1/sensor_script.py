import time
import csv
import paho.mqtt.client as mqtt



THINGSBOARD_HOST = 'localhost'  
ACCESS_TOKEN_AIR_HUM = '6magapt4zhx5nghwnwjz' 
ACCESS_TOKEN_PRESSURE = 'hrCzpND14sD272z2v9TY'
CSV_FILE = 'sensor_data.csv'

client_air_hum = mqtt.Client()
client_air_hum.username_pw_set(ACCESS_TOKEN_AIR_HUM)
client_air_hum.connect(THINGSBOARD_HOST, 1883, 60)

client_pressure = mqtt.Client()
client_pressure.username_pw_set(ACCESS_TOKEN_PRESSURE)
client_pressure.connect(THINGSBOARD_HOST, 1883, 60)

client_air_hum.loop_start()
client_pressure.loop_start()

try:
    with open(CSV_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            air_temperature = float(row['T (degC)'])
            humidity = float(row['rh (%)'])
            pressure = float(row['p (mbar)'])

            air_hum_payload = f'{{"air_temperature": {air_temperature}, "humidity": {humidity}}}'

            pressure_payload = f'{{"pressure": {pressure}}}'

            client_air_hum.publish('v1/devices/me/telemetry', air_hum_payload)
            client_pressure.publish('v1/devices/me/telemetry', pressure_payload)

            print(f'Sent air and humidity data: {air_hum_payload}')
            print(f'Sent pressure data: {pressure_payload}')

            time.sleep(5) 

except KeyboardInterrupt:
    pass

client_air_hum.loop_stop()
client_pressure.loop_stop()
client_air_hum.disconnect()
client_pressure.disconnect()