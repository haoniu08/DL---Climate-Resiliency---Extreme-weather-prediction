import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report
import requests
import json

"""
This class provides a simple implementation of a weather prediction system, which combines
LSTM-based sequence prediction with Random Forest-based current condition prediction.

The system is designed to predict extreme weather conditions based on historical weather data, which is fetched 
from WeatherStat API, and current weather conditions. The system trains two models - an LSTM model for sequence
prediction and a Random Forest model for current condition prediction. The predictions from both models are then
combined to generate an overall prediction, along with a confidence score.
"""

class ExtremeWeatherPrediction:
    def __init__(self, config=None):
        self.config = {
            'sequence_length': 30,  # Days of historical data to consider
            'lstm_units': 64,
            'dense_units': 32,
            'dropout_rate': 0.2,
            'rf_n_estimators': 100,
            'rf_max_depth': 15,
            'threshold_temperature': 35,  # Celsius
            'threshold_wind_speed': 100,  # km/h
            'threshold_precipitation': 100  # mm/day
        } if config is None else config
        
        self.lstm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        
    def fetch_weather_data(self, api_key, location, start_date, end_date):
        """
        Fetch weather data from WeatherStat API
        Note: This is a placeholder - implement actual API calls based on provider
        """
        try:
            # Example API endpoint
            url = f"https://api.weatherstat.com/historical"
            params = {
                'api_key': api_key,
                'location': location,
                'start_date': start_date,
                'end_date': end_date
            }
            response = requests.get(url, params=params)
            return pd.DataFrame(response.json()['data'])
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def prepare_sequences(self, data):
        """Prepare sequences for LSTM model"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.config['sequence_length']):
            sequence = data[i:(i + self.config['sequence_length'])]
            target = self._is_extreme_weather(data.iloc[i + self.config['sequence_length']])
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def _is_extreme_weather(self, weather_data):
        """Define extreme weather conditions"""
        return int(
            weather_data['temperature'] > self.config['threshold_temperature'] or
            weather_data['wind_speed'] > self.config['threshold_wind_speed'] or
            weather_data['precipitation'] > self.config['threshold_precipitation']
        )
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for sequence prediction"""
        model = Sequential([
            LSTM(self.config['lstm_units'], 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units']),
            Dropout(self.config['dropout_rate']),
            Dense(self.config['dense_units'], activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def build_rf_model(self):
        """Build Random Forest model for current conditions"""
        return RandomForestClassifier(
            n_estimators=self.config['rf_n_estimators'],
            max_depth=self.config['rf_max_depth'],
            random_state=42
        )
    
    def train_models(self, historical_data):
        """Train both LSTM and RF models"""
        # Prepare data for LSTM
        X_seq, y_seq = self.prepare_sequences(historical_data)
        X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Train LSTM
        self.lstm_model = self.build_lstm_model(X_seq.shape[1:])
        self.lstm_model.fit(
            X_seq_train, y_seq_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
        )
        
        # Prepare data for RF
        X_rf = historical_data.drop('target', axis=1)
        y_rf = historical_data['target']
        X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
            X_rf, y_rf, test_size=0.2, random_state=42
        )
        
        # Train RF
        self.rf_model = self.build_rf_model()
        self.rf_model.fit(X_rf_train, y_rf_train)
        
        # Evaluate models
        lstm_pred = (self.lstm_model.predict(X_seq_test) > 0.5).astype(int)
        rf_pred = self.rf_model.predict(X_rf_test)
        
        return {
            'lstm_metrics': classification_report(y_seq_test, lstm_pred),
            'rf_metrics': classification_report(y_rf_test, rf_pred)
        }
    
    def predict_extreme_weather(self, current_conditions, historical_sequence):
        """
        Combine predictions from both models
        Returns probability of extreme weather and confidence score
        """
        # LSTM prediction based on sequence
        lstm_prob = self.lstm_model.predict(np.array([historical_sequence]))[0][0]
        
        # RF prediction based on current conditions
        rf_prob = self.rf_model.predict_proba([current_conditions])[0][1]
        
        # Combine predictions (weighted average)
        combined_prob = 0.6 * lstm_prob + 0.4 * rf_prob
        
        # Calculate confidence score based on model agreement
        confidence = 1 - abs(lstm_prob - rf_prob)
        
        return {
            'probability': combined_prob,
            'confidence': confidence,
            'lstm_prob': lstm_prob,
            'rf_prob': rf_prob
        }
    
    def generate_alert(self, prediction_results, location):
        """Generate alert based on prediction results"""
        if prediction_results['probability'] > 0.7:
            alert_level = 'HIGH'
        elif prediction_results['probability'] > 0.4:
            alert_level = 'MEDIUM'
        else:
            alert_level = 'LOW'
            
        return {
            'location': location,
            'alert_level': alert_level,
            'probability': prediction_results['probability'],
            'confidence': prediction_results['confidence'],
            'timestamp': pd.Timestamp.now(),
            'recommendations': self._generate_recommendations(
                alert_level, 
                prediction_results
            )
        }
    
    def _generate_recommendations(self, alert_level, prediction_results):
        """Generate recommendations based on alert level and predictions"""
        recommendations = []
        
        if alert_level == 'HIGH':
            recommendations.extend([
                "Activate emergency response protocols",
                "Alert local emergency services",
                "Prepare for potential evacuation"
            ])
        elif alert_level == 'MEDIUM':
            recommendations.extend([
                "Monitor conditions closely",
                "Review emergency procedures",
                "Check emergency supplies"
            ])
            
        return recommendations