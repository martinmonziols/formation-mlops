import mlflow

model_name = "fasttext"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

list_libs = ["vendeur d'huitres", "boulanger", "COIFFEUR", "coiff5ur", "instagrammeur"]

test_data = {
    "query": list_libs,
    "k": 3
}

results = model.predict(test_data)
print(results)
