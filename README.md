# Despliegue de Modelo de Machine Learning - California Housing

API desarrollada con **FastAPI** y **Scikit-Learn** para predecir precios de viviendas en California, desplegada en **Render**.

## Pasos de Despliegue
1. **Cuenta y Repositorio:** Creación de cuenta en Render y configuración de un repositorio público en GitHub con el código fuente y el modelo entrenado (`modelo_california.pkl`).
2. **Dependencias:** Definición de librerías esenciales en el archivo `requirements.txt` (`fastapi`, `uvicorn`, `scikit-learn`, `pandas`, `joblib`).
3. **Comando de Inicio:** Configuración del servicio web en Render con el comando de arranque:
   `python -m uvicorn main:app --host 0.0.0.0 --port $PORT`

## Enlace Público
* **URL de la API:** https://dawm2d-opia-p2.onrender.com
* **Documentación (Swagger):** https://dawm2d-opia-p2.onrender.com/docs
