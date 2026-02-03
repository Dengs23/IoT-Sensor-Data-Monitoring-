# producer.py - Send IoT data to Event Hubs
import asyncio
import json
import random
from datetime import datetime
from azure.eventhub import EventData
from azure.eventhub.aio import EventHubProducerClient
import pandas as pd
import numpy as np

class IoTDataGenerator:
    def __init__(self):
        self.devices = [
            {'id': 'sensor_001', 'location': 'factory_a', 'type': 'temperature'},
            {'id': 'sensor_002', 'location': 'factory_a', 'type': 'humidity'},
            {'id': 'sensor_003', 'location': 'factory_b', 'type': 'pressure'},
            {'id': 'sensor_004', 'location': 'warehouse', 'type': 'motion'},
            {'id': 'sensor_005', 'location': 'warehouse', 'type': 'temperature'},
        ]
        
    def generate_sensor_data(self, device):
        """Generate realistic IoT sensor data"""
        base_values = {
            'temperature': {'mean': 22, 'std': 5, 'unit': '°C'},
            'humidity': {'mean': 45, 'std': 10, 'unit': '%'},
            'pressure': {'mean': 1013, 'std': 5, 'unit': 'hPa'},
            'motion': {'mean': 0.3, 'std': 0.1, 'unit': 'binary'}
        }
        
        sensor_type = device['type']
        config = base_values[sensor_type]
        
        # Generate value with occasional anomalies
        if sensor_type == 'motion':
            value = 1 if random.random() < config['mean'] else 0
        else:
            value = np.random.normal(config['mean'], config['std'])
            
            # Add anomaly with 5% probability
            if random.random() < 0.05:
                value *= random.uniform(1.5, 3.0)  # Spike anomaly
        
        return {
            'device_id': device['id'],
            'sensor_type': sensor_type,
            'value': round(value, 2),
            'unit': config['unit'],
            'location': device['location'],
            'timestamp': datetime.utcnow().isoformat(),
            'battery_level': round(random.uniform(30, 100), 2),
            'signal_strength': round(random.uniform(0.7, 1.0), 2)
        }

class EventHubProducer:
    def __init__(self, connection_string, eventhub_name):
        self.connection_string = connection_string
        self.eventhub_name = eventhub_name
        
    async def send_iot_data(self, num_messages=100, interval=1):
        """Send IoT data to Event Hub"""
        producer = EventHubProducerClient.from_connection_string(
            conn_str=self.connection_string,
            eventhub_name=self.eventhub_name
        )
        
        generator = IoTDataGenerator()
        
        try:
            for i in range(num_messages):
                # Create a batch
                event_data_batch = await producer.create_batch()
                
                # Add 5-10 events per batch (one per device)
                for device in generator.devices:
                    sensor_data = generator.generate_sensor_data(device)
                    event_data = EventData(json.dumps(sensor_data))
                    event_data.properties = {
                        'sensor_type': sensor_data['sensor_type'],
                        'location': sensor_data['location']
                    }
                    
                    try:
                        event_data_batch.add(event_data)
                    except ValueError:
                        # Batch is full, send it and create new
                        await producer.send_batch(event_data_batch)
                        event_data_batch = await producer.create_batch()
                        event_data_batch.add(event_data)
                
                # Send the batch
                await producer.send_batch(event_data_batch)
                print(f"Batch {i+1}/{num_messages} sent with {len(generator.devices)} events")
                
                # Wait before next batch
                await asyncio.sleep(interval)
                
        finally:
            await producer.close()

# Interactive usage
if __name__ == "__main__":
    import sys
    
    # Configuration
    CONNECTION_STR = "Endpoint=sb://your-namespace.servicebus.windows.net/;SharedAccessKeyName=..."
    EVENTHUB_NAME = "iot-sensor-data"
    
    producer = EventHubProducer(CONNECTION_STR, EVENTHUB_NAME)
    
    # Command line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == "send":
            num_messages = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            asyncio.run(producer.send_iot_data(num_messages))
        elif sys.argv[1] == "simulate":
            # Continuous simulation
            print("Starting continuous IoT data simulation...")
            while True:
                asyncio.run(producer.send_iot_data(10, 5))  # Send 10 batches every 5 seconds
    else:
        print("Usage: python producer.py [send|simulate] [num_messages]")
        
        
