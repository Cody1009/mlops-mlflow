echo mlflow_path $MLFLOW_BASE_PATH
mlflow server --backend-store-uri $MLFLOW_BASE_PATH/mlruns --host 0.0.0.0 --port 5000
