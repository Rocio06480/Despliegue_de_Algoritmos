import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from functions import entrenar_modelo_vinos

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trees", type=int, default=100)
    parser.add_argument("--experiment_name", type=str, default="Practica_Vinos")
    args = parser.parse_args()


    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment(args.experiment_name)


    with mlflow.start_run():
        model = entrenar_modelo_vinos(args.n_trees)


        data = load_wine()
        acc = accuracy_score(data.target, model.predict(data.data))


        mlflow.log_param("n_trees", args.n_trees)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")


