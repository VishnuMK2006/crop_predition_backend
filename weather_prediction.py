import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import numpy as np

class SimpleCropPricePredictorXGB:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None

    def load_and_train(self):
        data_path = "D:/3rd Year/Projects/full stack/backend/Datasets/cleaned_crop_data1.csv"

        # Load dataset
        data = pd.read_csv(data_path)
        data.drop_duplicates(subset=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], keep='first', inplace=True)

        # Features & target
        features = data.select_dtypes(include=[float, int]).columns
        target = 'label'
        X = data[features]
        y = data[target]

        # Encode target labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train XGBoost multi-class classifier
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(encoder.classes_),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        self.model = model
        self.scaler = scaler
        self.encoder = encoder
        print("XGBoost model training completed.")

    def predict_top3(self, feature_dict):
        features_df = pd.DataFrame([feature_dict])
        scaled_features = self.scaler.transform(features_df)
        proba = self.model.predict_proba(scaled_features)[0]

        # Get top 3 crops
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_crops = [(self.encoder.inverse_transform([i])[0], round(proba[i], 3)) for i in top3_idx]
        return top3_crops

# Usage example
if __name__ == "__main__":
    predictor = SimpleCropPricePredictorXGB()
    predictor.load_and_train()

    # Example input
    sample_features = {
        'N': 80,
        'P': 35,
        'K': 25,
        'temperature': 20.5,
        
        'humidity': 61,
        'ph': 5.5,
        'rainfall': 88
    }

    result = predictor.predict_top3(sample_features)
    print("Top 3 predicted crops with probabilities:", result)
