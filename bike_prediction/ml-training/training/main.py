import mlflow
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler


def read_raw_data(root_path):
    '''
    read in data 
    '''
    train = pd.read_csv(f"{root_path}/train.csv", parse_dates=[0])
    test = pd.read_csv(f"{root_path}/test.csv", parse_dates=[0])
    return train, test


def create_date_features(x):
    x['datetime'] = pd.to_datetime(x['datetime'])  
    return pd.DataFrame({
        'hour': x['datetime'].dt.hour,
        'day': x['datetime'].dt.day,
        'month': x['datetime'].dt.month,
        'year': x['datetime'].dt.year
    })


def rmsle(y, y_pred):
    # sum up y and y_pred
    y = np.sum(y, axis=1)
    y_pred = np.sum(y_pred, axis=1)    
    # set negative predictions to zero
    y_pred[y_pred < 0] = 0
    return np.sqrt(metrics.mean_squared_log_error(y, y_pred))
                

if __name__ == '__main__':

    
    
    mlflow.set_tracking_uri(f"https://dtacs:ultrapassword@mlflow.datatactics.dev")

    # mlflow.set_experiment("bike_example") # only needed it executing with python main.py

 
    with mlflow.start_run():
        train, test = read_raw_data("./data")

        datetime_pipeline = make_pipeline(
                                            FunctionTransformer(create_date_features, validate=False),
                                            OneHotEncoder(handle_unknown='ignore')
                                            )


        transformer = make_column_transformer(
                                            (datetime_pipeline, ['datetime']),
                                            (StandardScaler(), ['temp', 'atemp', 'humidity', 'windspeed']),         
                                            (OneHotEncoder(handle_unknown='ignore'), ['season', 'weather']),
                                            ('passthrough', ['workingday', 'holiday'])
                                        )

        model_pipeline = make_pipeline(
                                        transformer, 
                                        PolynomialFeatures(), 
                                        MultiOutputRegressor(PoissonRegressor(max_iter=10000))
                                        )

        param_grid = {
                        'polynomialfeatures__degree': [1, 2], 
                        'multioutputregressor__estimator__alpha': [0.001, 0.01, 0.1, 1, 10]
                    }

        X_train = train.drop(['casual','registered', 'count'], axis=1)
        y_train = train[['casual','registered']]

        neg_rmsle_score = metrics.make_scorer(rmsle, greater_is_better=False)
        cv = GridSearchCV(
                            model_pipeline, 
                            param_grid, 
                            return_train_score=True, 
                            scoring=neg_rmsle_score, 
                            cv=5, 
                            n_jobs=4, 
                            refit=True,
                            verbose=1
                            )
        cv.fit(X_train, y_train)

        mlflow.log_params(cv.best_params_) 
        mlflow.sklearn.log_model(cv.best_estimator_, "model")
        mlflow.log_metric("best_cross_val_score", cv.best_score_)
    
