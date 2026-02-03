# ml_anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, clear_output
import pickle
import json

class IoTAnomalyML:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.results = {}
        
    def prepare_features(self, feature_cols=None):
        """Prepare features for ML"""
        if feature_cols is None:
            feature_cols = ['temperature', 'humidity', 'pressure', 'battery_level']
        
        # Select and scale features
        self.features = self.data[feature_cols].fillna(0)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.features)
        
        return self.X_scaled
    
    def train_models(self, contamination=0.05):
        """Train multiple anomaly detection models"""
        print("Training anomaly detection models...")
        
        # Isolation Forest
        print("1. Training Isolation Forest...")
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        iso_predictions = iso_forest.fit_predict(self.X_scaled)
        self.models['isolation_forest'] = iso_forest
        self.results['isolation_forest'] = iso_predictions
        
        # One-Class SVM
        print("2. Training One-Class SVM...")
        oc_svm = OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='auto'
        )
        svm_predictions = oc_svm.fit_predict(self.X_scaled)
        self.models['one_class_svm'] = oc_svm
        self.results['one_class_svm'] = svm_predictions
        
        # Local Outlier Factor
        print("3. Training Local Outlier Factor...")
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            n_jobs=-1
        )
        lof_predictions = lof.fit_predict(self.X_scaled)
        self.models['local_outlier_factor'] = lof
        self.results['local_outlier_factor'] = lof_predictions
        
        print("Training complete!")
        
    def evaluate_models(self, true_labels=None):
        """Evaluate and compare models"""
        if true_labels is None:
            # Simulate true labels (in reality, would have labeled data)
            # Mark extreme values as anomalies
            true_labels = np.zeros(len(self.data))
            for i, col in enumerate(['temperature', 'humidity', 'pressure']):
                col_data = self.data[col].values
                q1, q3 = np.percentile(col_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                anomalies = (col_data < lower_bound) | (col_data > upper_bound)
                true_labels[anomalies] = 1
        
        evaluation_results = {}
        
        for model_name, predictions in self.results.items():
            # Convert predictions to binary (1 = normal, -1 = anomaly -> convert to 0 = normal, 1 = anomaly)
            binary_preds = (predictions == -1).astype(int)
            
            # Calculate metrics
            tp = np.sum((binary_preds == 1) & (true_labels == 1))
            fp = np.sum((binary_preds == 1) & (true_labels == 0))
            tn = np.sum((binary_preds == 0) & (true_labels == 0))
            fn = np.sum((binary_preds == 0) & (true_labels == 1))
            
            accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
                'anomaly_count': int(np.sum(binary_preds))
            }
        
        return evaluation_results
    
    def create_interactive_visualization(self):
        """Create interactive visualization of anomaly detection"""
        
        # Create widget for model selection
        model_selector = widgets.Dropdown(
            options=list(self.models.keys()),
            value='isolation_forest',
            description='Model:'
        )
        
        feature_x = widgets.Dropdown(
            options=self.features.columns.tolist(),
            value='temperature',
            description='X Axis:'
        )
        
        feature_y = widgets.Dropdown(
            options=self.features.columns.tolist(),
            value='humidity',
            description='Y Axis:'
        )
        
        output = widgets.Output()
        
        def update_plot(change):
            with output:
                clear_output(wait=True)
                
                model_name = model_selector.value
                x_feature = feature_x.value
                y_feature = feature_y.value
                
                # Get predictions for selected model
                predictions = self.results[model_name]
                is_anomaly = predictions == -1
                
                # Create scatter plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f'Anomaly Detection ({model_name})',
                        'Anomaly Distribution by Device',
                        f'{x_feature} Distribution',
                        f'{y_feature} Distribution'
                    )
                )
                
                # Main scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=self.data[x_feature][~is_anomaly],
                        y=self.data[y_feature][~is_anomaly],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='blue', size=8, opacity=0.6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.data[x_feature][is_anomaly],
                        y=self.data[y_feature][is_anomaly],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(color='red', size=12, opacity=0.8)
                    ),
                    row=1, col=1
                )
                
                # Anomalies by device
                if 'device_id' in self.data.columns:
                    device_anomalies = self.data[is_anomaly]['device_id'].value_counts()
                    fig.add_trace(
                        go.Bar(
                            x=device_anomalies.index,
                            y=device_anomalies.values,
                            name='Anomalies by Device'
                        ),
                        row=1, col=2
                    )
                
                # Distribution plots
                fig.add_trace(
                    go.Histogram(
                        x=self.data[x_feature],
                        nbinsx=30,
                        name=f'{x_feature} Distribution',
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Histogram(
                        x=self.data[y_feature],
                        nbinsx=30,
                        name=f'{y_feature} Distribution',
                        marker_color='lightgreen'
                    ),
                    row=2, col=2
                )
                
                # Add anomaly regions to distributions
                if is_anomaly.any():
                    # Calculate bounds for anomalies
                    x_anomalies = self.data[x_feature][is_anomaly]
                    y_anomalies = self.data[y_feature][is_anomaly]
                    
                    # Add vertical/horizontal lines for anomaly ranges
                    fig.add_vline(
                        x=np.percentile(x_anomalies, 90),
                        line_dash="dash",
                        line_color="red",
                        row=2, col=1
                    )
                    
                    fig.add_vline(
                        x=np.percentile(y_anomalies, 90),
                        line_dash="dash",
                        line_color="red",
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, showlegend=True)
                fig.show()
                
                # Show statistics
                anomaly_count = np.sum(is_anomaly)
                total_count = len(is_anomaly)
                
                print(f"Model: {model_name}")
                print(f"Total samples: {total_count}")
                print(f"Anomalies detected: {anomaly_count} ({anomaly_count/total_count:.1%})")
                
                if 'device_id' in self.data.columns:
                    print("\nTop devices with anomalies:")
                    device_stats = self.data[is_anomaly]['device_id'].value_counts().head(5)
                    for device, count in device_stats.items():
                        print(f"  {device}: {count} anomalies")
        
        # Wire up events
        model_selector.observe(update_plot, names='value')
        feature_x.observe(update_plot, names='value')
        feature_y.observe(update_plot, names='value')
        
        # Display widgets
        controls = widgets.VBox([model_selector, feature_x, feature_y])
        display(widgets.HBox([controls, output]))
        
        # Initial plot
        update_plot(None)

# Interactive usage example
def interactive_ml_demo():
    """Interactive ML anomaly detection demo"""
    
    # Load or generate data
    print("Generating IoT sensor data...")
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
        'device_id': np.random.choice([f'sensor_{i:03d}' for i in range(1, 11)], n_samples),
        'temperature': np.random.normal(22, 8, n_samples),
        'humidity': np.random.normal(45, 15, n_samples),
        'pressure': np.random.normal(1013, 10, n_samples),
        'battery_level': np.random.uniform(20, 100, n_samples)
    })
    
    # Add some synthetic anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data.loc[anomaly_indices, 'temperature'] *= np.random.uniform(1.5, 3.0, len(anomaly_indices))
    data.loc[np.random.choice(anomaly_indices, size=len(anomaly_indices)//2), 'humidity'] *= 0.3
    
    print(f"Generated {n_samples} samples with {len(anomaly_indices)} synthetic anomalies")
    
    # Initialize ML pipeline
    ml_pipeline = IoTAnomalyML(data)
    
    # Prepare features
    X = ml_pipeline.prepare_features()
    print(f"Prepared features with shape: {X.shape}")
    
    # Train models
    ml_pipeline.train_models(contamination=0.05)
    
    # Create interactive visualization
    print("\nLaunching interactive visualization...")
    ml_pipeline.create_interactive_visualization()
    
    # Evaluate models
    print("\nModel Evaluation:")
    evaluation = ml_pipeline.evaluate_models()
    
    # Display evaluation results
    eval_df = pd.DataFrame(evaluation).T
    print("\nPerformance Metrics:")
    print(eval_df[['accuracy', 'precision', 'recall', 'f1_score', 'anomaly_count']])
    
    return ml_pipeline

# Run the demo
if __name__ == "__main__":
    ml_pipeline = interactive_ml_demo()
    
    
