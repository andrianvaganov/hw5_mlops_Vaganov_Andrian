import os
import mlflow

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://147.45.147.94:9000'  #< - просто складывает на S3
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://0.0.0.0:9000'  #< - просто складывает на S3

os.environ['AWS_ACCESS_KEY_ID'] = 'mlflow'                      #< - просто складывает на S3
os.environ['AWS_SECRET_ACCESS_KEY'] = 'password'                #< - просто складывает на S3
#os.environ['MLFLOW_TRACKING_URI'] = 'http://147.45.147.94:5000'     #< - и хранит логи
os.environ['MLFLOW_TRACKING_URI'] = 'http://0.0.0.0:5000'     #< - и хранит логи

import mlflow
import mlflow.sklearn                                           #< - умеет работать с sklearn
from mlflow.models import infer_signature                       #< - умеет вычитывать сигнатуру функции

# mlflow.set_tracking_uri(uri="http://147.45.147.94:5000")
mlflow.set_tracking_uri(uri="http://0.0.0.0:5000")

mlflow.set_experiment("KPI tracing")
mlflow.autolog()                                                #< - и хранит логи

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

@mlflow.trace(name = "отображение трассировки", attributes = { "key": "value" })
def demonstrate_trace(inp=1): return 1

with mlflow.start_run(run_name="mlflow.start_run") as run:
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    params = {"max_depth": 10, "random_state": 42}

    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    mlflow.log_params(params)                                   #< - и хранит логи
    mlflow.log_metrics({"mse": mean_squared_error(y_test,       #< - и хранит логи
                                                  y_pred)})

    dataset = mlflow.data.from_pandas(df,
                                    source=dataset_source_url,
                                    name="это набор данных",    #< - просто складывает на S3
                                    targets='Appliances')
    mlflow.log_input(dataset, context="training")               #< - просто складывает на S3

    mlflow.sklearn.log_model(
        sk_model=model,
        name="sklearn-model",
        signature=signature,
        registered_model_name="это название модели",            #< - просто складывает на S3
    )

demonstrate_trace()