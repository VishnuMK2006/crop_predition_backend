import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Datasets', 'cleaned_crop_data1.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

print('Loading data from', DATA_PATH)
data = pd.read_csv(DATA_PATH)
data.drop_duplicates(subset=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], keep='first', inplace=True)

features = data.select_dtypes(include=[float, int]).columns
X = data[features]
y = data['label']

print('Encoding labels and scaling features...')
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Training XGBoost model...')
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
)
model.fit(X_scaled, y_encoded)

model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
encoder_path = os.path.join(MODELS_DIR, 'encoder.pkl')

print('Saving models to:')
print('  ', model_path)
print('  ', scaler_path)
print('  ', encoder_path)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(encoder, encoder_path)

print('Done.')
