# function_app.py - Real-time anomaly detection
import azure.functions as func
import json
import logging
import numpy as np
from datetime import datetime
from typing import List
import pandas as pd
from scipy import stats

app = func.FunctionApp()

# Store recent readings for each device (in-memory cache)
device_history = {}

class AnomalyDetector:
    """Real-time anomaly detection for IoT data"""
    
    @staticmethod
    def detect_zscore(value, history, threshold=3.0):
        """Detect anomaly using Z-score"""
        if len(history) < 10:
            return False, 0.0
        
        zscore = (value - np.mean(history)) / np.std(history)
        return abs(zscore) > threshold, zscore
    
    @staticmethod
    def detect_iqr(value, history):
        """Detect anomaly using Interquartile Range"""
        if len(history) < 20:
            return False, 0.0
        
        q1, q3 = np.percentile(history, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        is_anomaly = value < lower_bound or value > upper_bound
        return is_anomaly, value
    
    @staticmethod
    def detect_moving_average(value, history, window=10, threshold=2.0):
        """Detect anomaly using moving average"""
        if len(history) < window:
            return False, 0.0
        
        recent_values = history[-window:]
        moving_avg = np.mean(recent_values)
        moving_std = np.std(recent_values)
        
        if moving_std == 0:
            return False, 0.0
        
        deviation = abs(value - moving_avg) / moving_std
        return deviation > threshold, deviation

@app.event_hub_message_trigger(
    arg_name="events",
    event_hub_name="iot-sensor-data",
    connection="EventHubConnection"
)
def process_iot_data(events: List[func.EventHubEvent]):
    """Process IoT sensor data in real-time"""
    anomalies = []
    
    for event in events:
        try:
            # Parse the event
            sensor_data = json.loads(event.get_body().decode('utf-8'))
            device_id = sensor_data['device_id']
            value = sensor_data['value']
            timestamp = sensor_data['timestamp']
            
            # Initialize history for device if not exists
            if device_id not in device_history:
                device_history[device_id] = []
            
            # Add to history (keep last 100 readings)
            device_history[device_id].append(value)
            if len(device_history[device_id]) > 100:
                device_history[device_id] = device_history[device_id][-100:]
            
            # Run anomaly detection
            is_anomaly_z, zscore = AnomalyDetector.detect_zscore(
                value, device_history[device_id], threshold=3.0
            )
            
            is_anomaly_iqr, _ = AnomalyDetector.detect_iqr(
                value, device_history[device_id]
            )
            
            is_anomaly_ma, deviation = AnomalyDetector.detect_moving_average(
                value, device_history[device_id], window=10, threshold=2.5
            )
            
            # If any detection method flags anomaly
            is_anomaly = is_anomaly_z or is_anomaly_iqr or is_anomaly_ma
            
            if is_anomaly:
                anomaly_record = {
                    'device_id': device_id,
                    'sensor_value': value,
                    'unit': sensor_data['unit'],
                    'location': sensor_data['location'],
                    'timestamp': timestamp,
                    'detected_at': datetime.utcnow().isoformat(),
                    'detection_methods': {
                        'zscore': is_anomaly_z,
                        'iqr': is_anomaly_iqr,
                        'moving_average': is_anomaly_ma
                    },
                    'metrics': {
                        'zscore': float(zscore),
                        'deviation': float(deviation)
                    },
                    'history_size': len(device_history[device_id])
                }
                
                anomalies.append(anomaly_record)
                
                # Log the anomaly
                logging.warning(f"ANOMALY DETECTED: Device {device_id} - Value: {value}")
                
                # Send to Service Bus for alerting
                send_alert(anomaly_record)
        
        except Exception as e:
            logging.error(f"Error processing event: {str(e)}")
    
    # Return summary
    return {
        'processed_events': len(events),
        'anomalies_detected': len(anomalies),
        'anomaly_details': anomalies,
        'timestamp': datetime.utcnow().isoformat()
    }

def send_alert(anomaly_record):
    """Send alert to Service Bus (placeholder)"""
    # In production, would send to Service Bus for notification systems
    logging.info(f"Alert would be sent for anomaly: {anomaly_record['device_id']}")
    
@app.route(route="dashboard", auth_level=func.AuthLevel.ANONYMOUS)
def get_dashboard(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP endpoint for dashboard data"""
    # Calculate statistics
    stats = {}
    for device_id, history in device_history.items():
        if history:
            stats[device_id] = {
                'current': history[-1] if history else 0,
                'mean': float(np.mean(history)),
                'std': float(np.std(history)),
                'min': float(np.min(history)),
                'max': float(np.max(history)),
                'history_count': len(history)
            }
    
    response_data = {
        'device_statistics': stats,
        'total_devices_monitored': len(device_history),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return func.HttpResponse(
        json.dumps(response_data, indent=2),
        mimetype="application/json"
    )
    
    
