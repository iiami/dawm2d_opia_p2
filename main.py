import joblib
import numpy as np
import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API California Housing Model")

MODEL_PATH = "modelo_california.pkl"
GITHUB_MODEL_URL = "https://raw.githubusercontent.com/iiami/dawm2d_opia_p2/main/modelo_california.pkl"

if not os.path.exists(MODEL_PATH):
  print("El modelo no está en local. Descargándolo desde GitHub...")
  response = requests.get(GITHUB_MODEL_URL)
  if response.status_code == 200:
    with open(MODEL_PATH, "wb") as f:
      f.write(response.content)
    print("Modelo descargado correctamente.")
  else:
    raise Exception(
        f"No se pudo descargar el modelo. Código HTTP: {response.status_code}"
    )

modelo = joblib.load(MODEL_PATH)


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