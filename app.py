from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timedelta
from unified_prediction import get_unified_prediction


app = Flask(__name__)
CORS(app)

class CropRecommendationSystem:
    def __init__(self):
        self.price_model = None
        self.crop_model = None
        self.scaler = None
        self.encoder = None
        self.avg_price = None
        self.crops_to_forecast = ['Banana', 'Coconut', 'Coffee', 'Cotton', 'Maize', 'Rice']
        self.weather_base_url = 'https://api.open-meteo.com/v1/forecast'
        
    def load_price_data(self):
        """Load and prepare price prediction data"""
        try:
            df = pd.read_csv("Datasets/crop_price.csv")
            df = df.dropna()
            df['Price Date'] = pd.to_datetime(df['Price Date'], format='%b-%y')
            df['Month'] = df['Price Date'].dt.month
            df['Year'] = df['Price Date'].dt.year
            self.avg_price = df.groupby(['District', 'Crop', 'Year', 'Month'])['Crop Price (Rs per quintal)'].mean().reset_index()
            return True
        except Exception as e:
            print(f"Error loading price data: {e}")
            return False
    
    def load_crop_model(self):
        """Load pre-trained crop recommendation model"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, 'models')
            model_path = os.path.join(models_dir, 'xgb_model.pkl')
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            encoder_path = os.path.join(models_dir, 'encoder.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
                self.crop_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.encoder = joblib.load(encoder_path)
                return True
            else:
                # Train new model in-memory if pre-trained doesn't exist
                return self.train_crop_model()
        except Exception as e:
            print(f"Error loading crop model: {e}")
            return self.train_crop_model()
    
    def train_crop_model(self):
        """Train crop recommendation model"""
        try:
            data = pd.read_csv("Datasets/cleaned_crop_data1.csv")
            data.drop_duplicates(subset=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], keep='first', inplace=True)
            
            features = data.select_dtypes(include=[float, int]).columns
            target = 'label'
            X = data[features]
            y = data[target]
            
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(encoder.classes_),
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42
            )
            model.fit(X_scaled, y_encoded)
            
            self.crop_model = model
            self.scaler = scaler
            self.encoder = encoder
            return True
        except Exception as e:
            print(f"Error training crop model: {e}")
            return False
    
    def get_weather_data(self, lat, lon, days=14):
        """Fetch weather forecast from Open-Meteo API"""
        try:
            # Ensure days is within valid range (1-14)
            days = max(1, min(14, int(days)))
            
            # Open-Meteo API parameters for weather forecast
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max,relative_humidity_2m_min",
                "timezone": "auto",
                "forecast_days": days
            }
            
            response = requests.get(self.weather_base_url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code != 200:
                raise Exception(f"Weather API error: {data.get('error', 'Unknown error')}")
            
            # Process current weather (first hour of data)
            current_time = datetime.now()
            hourly_data = data['hourly']
            timezone_offset = data['utc_offset_seconds']
            
            # Find current hour data
            current_hour_data = {
                'temperature': hourly_data['temperature_2m'][0],
                'humidity': hourly_data['relative_humidity_2m'][0],
                'pressure': hourly_data['pressure_msl'][0],
                'wind_speed': hourly_data['wind_speed_10m'][0],
                'precipitation': hourly_data['precipitation'][0],
                'timestamp': current_time.isoformat()
            }
            
            # Process daily forecast data
            daily_data = data['daily']
            processed_forecast = []
            
            for i in range(min(days, len(daily_data['time']))):
                date_str = daily_data['time'][i]
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                
                # Calculate average temperature from max and min
                avg_temp = (daily_data['temperature_2m_max'][i] + daily_data['temperature_2m_min'][i]) / 2
                avg_humidity = (daily_data['relative_humidity_2m_max'][i] + daily_data['relative_humidity_2m_min'][i]) / 2
                
                processed_forecast.append({
                    'date': date_str,
                    'temperature': round(avg_temp, 2),
                    'humidity': round(avg_humidity, 2),
                    'pressure': round(daily_data['pressure_msl'][i] if 'pressure_msl' in daily_data else 1013.25, 2),
                    'wind_speed': round(daily_data['wind_speed_10m'][i] if 'wind_speed_10m' in daily_data else 0, 2),
                    'rainfall': round(daily_data['precipitation_sum'][i], 2)
                })
            
            # Get current weather description based on temperature and precipitation
            temp = current_hour_data['temperature']
            precip = current_hour_data['precipitation']
            
            if precip > 0.1:
                description = "Rainy" if precip > 2 else "Light rain"
            elif temp > 25:
                description = "Sunny" if temp > 30 else "Partly cloudy"
            elif temp < 10:
                description = "Cold" if temp < 5 else "Cool"
            else:
                description = "Mild"
            
            current_weather = {
                'temperature': current_hour_data['temperature'],
                'humidity': current_hour_data['humidity'],
                'pressure': current_hour_data['pressure'],
                'description': description,
                'wind_speed': current_hour_data['wind_speed'],
                'precipitation': current_hour_data['precipitation'],
                'timestamp': current_time.isoformat()
            }
            
            return {
                'current': current_weather,
                'forecast': processed_forecast,
                'location': {
                    'lat': lat,
                    'lon': lon,
                    'timezone': data.get('timezone', 'Unknown'),
                    'timezone_abbreviation': data.get('timezone_abbreviation', 'Unknown')
                }
            }
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_for_crop_recommendation(self, lat, lon, days=7):
        """Get weather data specifically formatted for crop recommendation"""
        weather_data = self.get_weather_data(lat, lon, days)
        if not weather_data:
            return None
        
        # Use current weather for immediate recommendation
        current = weather_data['current']
        
        # Calculate average conditions for the specified number of days
        forecast = weather_data['forecast'][:days]
        
        if len(forecast) > 0:
            avg_temp = sum(day['temperature'] for day in forecast) / len(forecast)
            avg_humidity = sum(day['humidity'] for day in forecast) / len(forecast)
            total_rainfall = sum(day['rainfall'] for day in forecast)
        else:
            # Fallback to current weather if no forecast available
            avg_temp = current['temperature']
            avg_humidity = current['humidity']
            total_rainfall = current.get('precipitation', 0)
        
        return {
            'temperature': round(avg_temp, 2),
            'humidity': round(avg_humidity, 2),
            'rainfall': round(total_rainfall, 2),
            'current_weather': current,
            'forecast_summary': {
                'avg_temperature': round(avg_temp, 2),
                'avg_humidity': round(avg_humidity, 2),
                'total_rainfall': round(total_rainfall, 2),
                'forecast_days': len(forecast)
            }
        }

# Initialize the system
crop_system = CropRecommendationSystem()
crop_system.load_price_data()
crop_system.load_crop_model()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Crop Recommendation API is running"})

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """Get weather data for given coordinates"""
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        days = request.args.get('days', 14, type=int)
        
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        weather_data = crop_system.get_weather_data(lat, lon, days)
        
        if not weather_data:
            return jsonify({"error": "Failed to fetch weather data"}), 500
        
        return jsonify(weather_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/weather-for-crops', methods=['GET'])
def get_weather_for_crops():
    """Get weather data formatted for crop recommendation"""
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        days = request.args.get('days', 7, type=int)
        
        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        weather_data = crop_system.get_weather_for_crop_recommendation(lat, lon, days)
        
        if not weather_data:
            return jsonify({"error": "Failed to fetch weather data"}), 500
        
        return jsonify(weather_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/crop-recommendation', methods=['POST'])
def get_crop_recommendation():
    try:
        data = request.json
        
        # Check if location-based recommendation is requested
        if 'lat' in data and 'lon' in data:
            # Location-based recommendation using weather data
            lat = float(data['lat'])
            lon = float(data['lon'])
            days = int(data.get('days', 7))  # Default to 7 days
            
            # Get weather data
            weather_data = crop_system.get_weather_for_crop_recommendation(lat, lon, days)
            if not weather_data:
                return jsonify({"error": "Failed to fetch weather data for the given location"}), 500
            
            # Use default soil values if not provided
            features = {
                'N': float(data.get('N', 80)),  # Default nitrogen
                'P': float(data.get('P', 35)),  # Default phosphorus
                'K': float(data.get('K', 25)),  # Default potassium
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'ph': float(data.get('ph', 6.5)),  # Default pH
                'rainfall': weather_data['rainfall']
            }
            
            weather_info = {
                'location': f"{lat}, {lon}",
                'weather_source': 'Open-Meteo API',
                'current_weather': weather_data['current_weather'],
                'forecast_summary': weather_data['forecast_summary']
            }
        else:
            # Manual input recommendation
            required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            # Validate input
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            features = {
                'N': float(data['N']),
                'P': float(data['P']),
                'K': float(data['K']),
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
            
            weather_info = {
                'location': 'Manual Input',
                'weather_source': 'User Input'
            }
        
        # Load model if not already loaded
        if crop_system.crop_model is None:
            if not crop_system.load_crop_model():
                return jsonify({"error": "Failed to load crop recommendation model"}), 500
        
        # Define the feature order
        feature_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Make prediction
        features_df = pd.DataFrame([features])
        features_df = features_df[feature_order]  # Enforce column order
        scaled_features = crop_system.scaler.transform(features_df)
        proba = crop_system.crop_model.predict_proba(scaled_features)[0]
        
        # Get top 3 crops
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_crops = [
            {
                "crop": crop_system.encoder.inverse_transform([i])[0],
                "probability": float(proba[i])
            } 
            for i in top3_idx
        ]
        
        return jsonify({
            "recommendations": top3_crops,
            "input_conditions": features,
            "weather_info": weather_info
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/price-prediction', methods=['POST'])
def get_price_prediction():
    try:
        data = request.json
        
        if 'district' not in data:
            return jsonify({"error": "Missing required field: district"}), 400
        
        district = data['district']
        
        # Use the improved predict_for_district function with accuracy metrics
        from price_prediction import predict_for_district
        predictions = predict_for_district(district)
        
        # Format response for frontend compatibility
        result = {}
        for crop, crop_data in predictions.items():
            if 'predictions' in crop_data and crop_data['predictions'].get('month1', -1) != -1:
                result[crop] = {
                    "month1": crop_data['predictions']['month1'],
                    "month2": crop_data['predictions']['month2'],
                    "month3": crop_data['predictions']['month3'],
                    "accuracy": crop_data['accuracy']
                }
            else:
                result[crop] = {
                    "month1": "No data",
                    "month2": "No data", 
                    "month3": "No data",
                    "accuracy": None
                }
        
        return jsonify({
            "district": district,
            "predictions": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/available-districts', methods=['GET'])
def get_available_districts():
    try:
        if crop_system.avg_price is None:
            if not crop_system.load_price_data():
                return jsonify({"error": "Failed to load price data"}), 500
        
        districts = sorted(crop_system.avg_price['District'].unique().tolist())
        return jsonify({"districts": districts})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/unified-predict', methods=['POST'])
def unified_predict():
    """Unified prediction endpoint"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        # Extract soil and district data from the nested structure
        soil_data_raw = data.get('soil_data', {})
        soil_data = {
            'N': float(soil_data_raw.get('N')),
            'P': float(soil_data_raw.get('P')),
            'K': float(soil_data_raw.get('K')),
            'temperature': float(soil_data_raw.get('temperature')),
            'humidity': float(soil_data_raw.get('humidity')),
            'ph': float(soil_data_raw.get('ph')),
            'rainfall': float(soil_data_raw.get('rainfall'))
        }
        district = data.get('district')

        if not district:
            return jsonify({"error": "District is required"}), 400

        # Get unified prediction
        recommendation = get_unified_prediction(soil_data, district)

        if recommendation:
            return jsonify({"recommended_crop": recommendation})
        else:
            return jsonify({"message": "No suitable crop found"}), 404

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid data format: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    # Initialize and load all necessary components
    if crop_system.load_price_data() and crop_system.load_crop_model():
        print("Crop recommendation system loaded successfully.")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load crop recommendation system.")
