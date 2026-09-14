from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning
from pathlib import Path
import os

# set the output as pandas
set_config(transform_output='pandas')

class Data(BaseModel):  
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

    
    
# load the locally trained model pipeline
root_path = Path(__file__).parent
model_path = root_path / "models" / "model.joblib"
model_pipe = joblib.load(model_path)

# create the app
app = FastAPI()

# create the home endpoint
@app.get(path="/")
def home():
    return FileResponse(root_path / "frontend" / "index.html")

# create the predict endpoint
@app.post(path="/predict")
def do_predictions(data: Data):
    pred_data = pd.DataFrame({
        'ID': data.ID,
        'Delivery_person_ID': data.Delivery_person_ID,
        'Delivery_person_Age': data.Delivery_person_Age,
        'Delivery_person_Ratings': data.Delivery_person_Ratings,
        'Restaurant_latitude': data.Restaurant_latitude,
        'Restaurant_longitude': data.Restaurant_longitude,
        'Delivery_location_latitude': data.Delivery_location_latitude,
        'Delivery_location_longitude': data.Delivery_location_longitude,
        'Order_Date': data.Order_Date,
        'Time_Orderd': data.Time_Orderd,
        'Time_Order_picked': data.Time_Order_picked,
        'Weatherconditions': data.Weatherconditions,
        'Road_traffic_density': data.Road_traffic_density,
        'Vehicle_condition': data.Vehicle_condition,
        'Type_of_order': data.Type_of_order,
        'Type_of_vehicle': data.Type_of_vehicle,
        'multiple_deliveries': data.multiple_deliveries,
        'Festival': data.Festival,
        'City': data.City
        },index=[0]
    )
    # clean the raw input data
    cleaned_data = perform_data_cleaning(pred_data)
    if cleaned_data.empty:
        raise HTTPException(
            status_code=422,
            detail="Prediction input contains missing or invalid delivery data",
        )
    # get the predictions
    predictions = model_pipe.predict(cleaned_data)[0]

    return predictions
   
   
if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))