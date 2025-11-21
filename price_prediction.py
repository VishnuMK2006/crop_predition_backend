import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
# Load dataset with relative path
csv_path = os.path.join(os.path.dirname(__file__), 'Datasets', 'crop_price.csv')
df = pd.read_csv(csv_path)

# Data Cleaning
df = df.dropna()
df['Price Date'] = pd.to_datetime(df['Price Date'], format='%b-%y')

# Feature Engineering
df['Month'] = df['Price Date'].dt.month
df['Year'] = df['Price Date'].dt.year

# Average price per month
avg_price = df.groupby(['District', 'Crop', 'Year', 'Month'])['Crop Price (Rs per quintal)'].mean().reset_index()

# List of crops to forecast
crops_to_forecast = ['Banana', 'Coconut', 'Coffee', 'Cotton', 'Maize', 'Rice']

# Function to forecast prices with accuracy metrics
def forecast_prices(district, crop):
    subset = avg_price[(avg_price['District'] == district) & (avg_price['Crop'] == crop)]
    
    if len(subset) < 2:
        return {
            'prices': [-1],
            'accuracy': None
        }
    
    X_train = subset[['Month', 'Year']]
    y_train = subset['Crop Price (Rs per quintal)']

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate accuracy metrics on training data
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    
    # Calculate accuracy percentage (based on MAPE - Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100
    accuracy_percentage = max(0, 100 - mape)  # Convert MAPE to accuracy

    latest_month = subset[['Month', 'Year']].iloc[-1].copy()
    forecasted_prices = []

    for i in range(1, 4):
        latest_month['Month'] += 1
        if latest_month['Month'] > 12:
            latest_month['Month'] = 1
            latest_month['Year'] += 1

        next_month_data = pd.DataFrame([latest_month], columns=['Month', 'Year'])
        forecast = model.predict(next_month_data)
        forecasted_prices.append(round(forecast[0], 2))
    
    return {
        'prices': forecasted_prices,
        'accuracy': {
            'r2_score': round(r2, 4),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'accuracy_percentage': round(accuracy_percentage, 2)
        }
    }

# Main function to predict all crops for a district
def predict_for_district(district):
    print(f"=== Forecasted Prices for {district} ===\n")
    result = {}

    for crop in crops_to_forecast:
        forecast_data = forecast_prices(district, crop)
        prices = forecast_data['prices']
        converted = [float(x) for x in prices]

        # Map prices to month1, month2, month3
        month_map = {f"month{i+1}": val for i, val in enumerate(converted)}
        
        # Add accuracy metrics
        result[crop] = {
            'predictions': month_map,
            'accuracy': forecast_data['accuracy']
        }

    # Print JSON nicely
    print(json.dumps(result, indent=2))
    return result

# Example usage
if __name__ == "__main__":
    district_name = input("Enter District/City Name: ")
    predict_for_district(district_name)
    