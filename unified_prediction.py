import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from weather_prediction import SimpleCropPricePredictorXGB
from price_prediction import predict_for_district

def safe_float(val):
    """Convert numpy.float32, float64, or int to Python float safely."""
    try:
        return float(val)
    except Exception:
        return val

def get_unified_prediction(soil_data, district):
    """
    Provides a unified crop prediction based on the intersection of
    soil/weather-based crops and market price-based crops (case-insensitive).
    Also returns confidence score from weather model.
    """

    # === 1. Weather/soil-based prediction (with probabilities) ===
    crop_predictor = SimpleCropPricePredictorXGB()
    crop_predictor.load_and_train()
    top_crops = crop_predictor.predict_top3(soil_data)   # [(crop, prob), ...]
    weather_crops = {crop: safe_float(prob) for crop, prob in top_crops}  # dict for easy lookup

    # === 2. Market price-based prediction ===
    price_predictions = predict_for_district(district)
    market_crops = list(price_predictions.keys())

    # === 3. Case-insensitive intersection ===
    weather_lower = {c.lower(): c for c in weather_crops.keys()}
    market_lower = {c.lower(): c for c in market_crops}

    common_lower = set(weather_lower.keys()) & set(market_lower.keys())
    common_crops = [market_lower[c] for c in common_lower]   # keep market case

    # === 4. If intersection exists, choose crop with weighted score (70% confidence + 30% price) ===
    recommended_crop = None
    confidence_score = None
    best_score = -1

    if common_crops:
        # Calculate all prices and find min/max for normalization
        all_prices = {}
        for crop in common_crops:
            # Handle new format with 'predictions' and 'accuracy' keys
            crop_data = price_predictions[crop]
            if isinstance(crop_data, dict) and 'predictions' in crop_data:
                prices = [v for v in crop_data['predictions'].values() if isinstance(v, (int, float)) and v != -1]
            else:
                # Fallback for old format
                prices = [v for v in crop_data.values() if isinstance(v, (int, float)) and v != -1]
            
            if prices:
                all_prices[crop] = safe_float(sum(prices) / len(prices))
            else:
                all_prices[crop] = 0
        
        max_price = max(all_prices.values()) if all_prices else 1
        min_price = min(all_prices.values()) if all_prices else 0
        price_range = max_price - min_price if max_price > min_price else 1
        
        # Calculate weighted score for each common crop
        for crop in common_crops:
            # Get weather confidence (case-insensitive lookup)
            crop_lower = crop.lower()
            if crop_lower in weather_lower:
                original_crop = weather_lower[crop_lower]
                crop_confidence = safe_float(weather_crops.get(original_crop, 0))
            else:
                crop_confidence = 0
            
            # Normalize price to 0-1 scale
            normalized_price = (all_prices[crop] - min_price) / price_range if price_range > 0 else 0.5
            
            # Weighted score: 70% weather confidence + 30% price favorability
            score = (0.7 * crop_confidence) + (0.3 * normalized_price)
            
            if score > best_score:
                best_score = score
                recommended_crop = crop
                confidence_score = crop_confidence

    return {
        "weather_top3": [(c, safe_float(p)) for c, p in top_crops],  # force float
        "market_crops": market_crops,
        "common_crops": common_crops,
        "recommended_crop": recommended_crop,
        "confidence_score": safe_float(confidence_score) if confidence_score is not None else None
    }

if __name__ == '__main__':
    # Example usage
    sample_soil_data = {
        'N': 80,
        'P': 35,
        'K': 25,
        'temperature': 20.5,
        'humidity': 61,
        'ph': 5.5,
        'rainfall': 88
    }
    sample_district = "Nicobar"  # Example district

    unified_result = get_unified_prediction(sample_soil_data, sample_district)
    import json
    print(json.dumps(unified_result, indent=2))  # Now safe for JSON
