class MlflowConfig():
  PROJECT_PATH = "/var/lib/mlflow"
  BASE_PATH = "/var/lib/mlflow/artifacts"
  ARTIFACT_PATHS = {
    "input_data_dir": f"{BASE_PATH}/data/input/",
    "raw_data_path": f"{BASE_PATH}/data/input/raw-data.csv",
    "output_data_dir": f"{BASE_PATH}/data/output/",
    "model_dir": f"{BASE_PATH}/model/",
    "model_path": f"{BASE_PATH}/model/model.json",
    "importance_fig_path": f"{BASE_PATH}/model/importance.png",
  }
