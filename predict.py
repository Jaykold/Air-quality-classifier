from typing import Dict
import pickle

import warnings
warnings.filterwarnings("ignore")


model_path = "model/model.pkl"
scaler_path = "model/scaler.pkl"
le_path = "model/label_encoder.pkl"

def load_model():
    with open(model_path, 'rb') as model_file, \
         open(scaler_path, 'rb') as scaler_file, \
         open(le_path, 'rb') as le_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
        le = pickle.load(le_file)
    return model, scaler, le

def preprocess_data(data: Dict) -> Dict:
    if not isinstance(data, Dict):
        raise TypeError("Data must be a dictionary")
    
    if 'pm10' in data.keys():
        del data['pm10']
    X = list(data.values())
   
    return X

def predict(data: Dict):
    X = preprocess_data(data)
    model, scaler, le = load_model()
    
    data = scaler.transform([X])
    y_pred = model.predict(data)
    y_pred = le.inverse_transform(y_pred)[0]

    return y_pred

if __name__ == "__main__":
    data = {
    "temperature": 30.4,
    "humidity": 98.0,
    "pm2.5": 24.9,
    "pm10": 30.7,
    "no2": 23.5,
    "so2": 21.2,
    "co": 1.55,
    "proximity_to_industrial_areas": 6.7,
    "population_density": 594.0}
   
    predictions = predict(data)
    print(f"Predictions: {predictions}")