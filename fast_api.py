from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import Literal



model = joblib.load('model.pkl')

app = FastAPI(
    title='Titanic Survival Prediction',
    verbose='1.0',
    description='Predictiong the survival of passenger'

)

class Passanger(BaseModel):
    Pclass: Literal[1,2,3]
    Sex: Literal['Male','Female']
    Age: int = Field(..., ge = 0)
    Fare: float = Field(...,ge = 0)
    Cabin: str | None = None
    Embarked: Literal['Q','C','S']
    with_family: int = Field(...)



@app.get('/')
def home():
    return {
        'message':'ML prediction base system'
    }


@app.post('/predict')
def Passanger(passenger: Passanger):
    df = pd.DataFrame([passenger.model_dump()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return {
         "prediction": int(prediction),

            "survival": (
                "Survived"
                if prediction == 1
                else "Not Survived"
            ),

            "probability": {

                "Not Survived": round(float(probability[0]), 4),

                "Survived": round(float(probability[1]), 4)

            }
    }

