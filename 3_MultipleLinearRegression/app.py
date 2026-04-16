from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Lade das gespeicherte Modell
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Definiere das Datenformat für die Eingabe
class CarData(BaseModel):
    Mileage: float
    EngineV: float
    Brand_BMW: bool
    Brand_Mercedes_Benz: bool
    Brand_Mitsubishi: bool
    Brand_Renault: bool
    Brand_Toyota: bool
    Brand_Volkswagen: bool
    Body_hatch: bool
    Body_other: bool
    Body_sedan: bool
    Body_vagon: bool
    Body_van: bool
    Engine_Type_Gas: bool
    Engine_Type_Other: bool
    Engine_Type_Petrol: bool
    Registration_yes: bool

@app.post("/predict")
def predict_price(data: CarData):
    # Eingaben in das richtige Format für das Modell bringen
    features = np.array([[
        data.Mileage, data.EngineV, data.Brand_BMW, data.Brand_Mercedes_Benz,
        data.Brand_Mitsubishi, data.Brand_Renault, data.Brand_Toyota,
        data.Brand_Volkswagen, data.Body_hatch, data.Body_other,
        data.Body_sedan, data.Body_vagon, data.Body_van,
        data.Engine_Type_Gas, data.Engine_Type_Other,
        data.Engine_Type_Petrol, data.Registration_yes
    ]])
    
    # Skalierung der Features (falls du im Notebook skaliert hast, den Scaler hier laden und anwenden!)
    # features_scaled = scaler.transform(features)
    
    # Vorhersage des log_price
    log_prediction = model.predict(features) # (oder features_scaled verwenden)
    
    # Logarithmus rückgängig machen (Exponentialfunktion), um den echten Preis zu erhalten
    actual_price = np.exp(log_prediction[0])
    
    return {"predicted_price": actual_price}

@app.get("/")
def read_root():
    return {"message": "Willkommen bei der ML Car Price Prediction API!"}