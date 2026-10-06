import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. Cargar datos de California (sección 7.2 de la UD1)
california = fetch_california_housing(as_frame=True)
X, y = california.data, california.target

# 2. Dividir en entrenamiento y test (sección 3.5 / 7.2 de la UD1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Entrenar el modelo (sección 7.2 de la UD1)
modelo = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo.fit(X_train, y_train)

# 4. Guardar el modelo entrenado como exige el ejercicio (sección 7.6 de la UD1)
joblib.dump(modelo, "modelo_california.pkl")
print("¡Modelo generado y guardado correctamente como modelo_california.pkl!")