# Bike-Prediction-Model

This repository holds scripts relating to EDA and training of bike rental prediction model.

This is the QOD mlops cycle:

```mermaid
graph
    A[Train Model] --> B(Build and Push image)
    B --> C{Serve image}
    C --> D(Make call to model for predictions)
    D --> E[Get new data]
    E --> A
``` 

### Before interacting with MLFLOW ensure the ENVs are available:

```bash
export MLFLOW_TRACKING_URI="https://user:password@mlflow-app.de"  
```

## Train model using mlflow

```bash
 mlflow run training --experiment-name bike_example --build-image --entry-point main --backend kubernetes --backend-config dtacs.json
```

## Build Model

### generate docker file to server the model:

- manually generate the docker file and needed artifacts using this command
```bash
mlflow models generate-dockerfile -m runs:/7d2d006eb6e1403da25149849a9355ff/model
``` 

- build the model:
```bash
docker build . -t bike
```

- tag and push the model:
```bash
docker tag bike:latest docker.repo.name/bike:latest
docker push docker.repo.name/bike:latest
```

- or put run number in `.github/workflows/build.yaml` and the model will automatically be servered

## Serve model 

- use k8s to serve model in cluster
add link to servering manifests for bike example

## Predict

- Model being served on our cluster and can accept 'processed data' for a prediction. 
- Currently only secured with a simple basic-auth
- Example execution:
```bash
$  curl -X POST --user user:password https://mlflow-serving.datatactics.dev/invocations  \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {
        "datetime": "2011-07-13 06:00:00",
        "season": 3,
        "holiday": 0,
        "workingday": 1,
        "weather": 1,
        "temp": 30.34,
        "atemp": 34.09,
        "humidity": 58,
        "windspeed": 11.0014
      }
    ]
  }' 
```

## Re-train

- When there is new data or features that will help improve the model then it should be re-trained the cycle begins again

MLops is a continual cycle