import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API California Housing Model")

modelo = joblib.load("modelo_california.pkl")


class EntradaVariables(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def inicio():
    return {"mensaje": "API de predicción desplegada correctamente"}


@app.post("/predict")
def predecir(datos: EntradaVariables):
    entrada = np.array(
        [
            [
                datos.MedInc,
                datos.HouseAge,
                datos.AveRooms,
                datos.AveBedrms,
                datos.Population,
                datos.AveOccup,
                datos.Latitude,
                datos.Longitude,
            ]
        ]
    )

    prediccion = modelo.predict(entrada)
    return {"precio_estimado": float(prediccion[0])}