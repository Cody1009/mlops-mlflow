# sample codes from https://www.asigmo.com/post/mlflow-best-practices-and-lessons-learned

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from settings import MlflowConfig

class MlflowTest():
    def set_config(self):
        print('set_config')
        self.mlflow_config = MlflowConfig()
     
    def load_data(self):
        print('load_data')
        self.dataset = pd.read_csv(self.mlflow_config.ARTIFACT_PATHS['raw_data_path'], sep=';')
     
    def preprocess_data(self):
        print('preprocess_data')
        self.X = self.dataset.drop('quality', axis=1)
        y = self.dataset.loc[:, 'quality']
        y = y.replace({5: 0, 6: 1, 7: 2, 4: 3, 8: 4, 3: 5})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, random_state=110, stratify=y)

    def setup_mlflow(self):
        print('setup_mlflow')
        artifact_uri = f"file://{self.mlflow_config.PROJECT_PATH}/mlruns"
        mlflow.set_tracking_uri(artifact_uri)
        print(f"set tracking uri at {artifact_uri}")
        mlflow.set_experiment("xgboost-logging-demo")
        
    def generate_model(self):
        print('generate_model')
        scaler = MinMaxScaler()
        scaler.fit(self.X_train, self.y_train)
        X_train_scaled = scaler.transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        dtrain = xgb.DMatrix(X_train_scaled, label=self.y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=self.y_test)

        # enable auto logging
        mlflow.xgboost.autolog(log_input_examples=True)
        with mlflow.start_run():
            # train model
            params = {
                'objective': 'multi:softprob',
                'num_class': 6,
                'eval_metric': 'mlogloss',
                'colsample_bytree': 0.9,
                'subsample': 0.9,
                'seed': 6174,
            }
            model = xgb.train(params, dtrain, evals=[(dtrain, 'train')])
            # evaluate the model
            y_proba = model.predict(dtest)
            y_pred = y_proba.argmax(axis=1)
            loss = log_loss(self.y_test, y_proba)
            acc = accuracy_score(self.y_test, y_pred)

            # save outputs
            model.save_model(self.mlflow_config.ARTIFACT_PATHS['model_path'])
            importance_fig = xgb.plot_importance(model)
            importance_fig.figure.savefig(self.mlflow_config.ARTIFACT_PATHS['importance_fig_path'])

            # log metrics
            mlflow.log_metrics({'log_loss': loss, 'accuracy': acc})
            mlflow.log_artifacts(f"{self.mlflow_config.BASE_PATH}")

def main():
    c = MlflowTest()
    c.set_config()
    c.load_data()
    c.preprocess_data()
    c.setup_mlflow()
    c.generate_model()

if __name__ == '__main__':
    main()
