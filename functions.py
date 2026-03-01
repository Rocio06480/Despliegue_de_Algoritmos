from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

def entrenar_modelo_vinos(n_trees):
    data = load_wine()
    X, y = data.data, data.target


    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)


    model.fit(X, y)

    return model
