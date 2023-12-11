## Guide

### To run the server
```commandline
python flask_app.py
```


### To enquire about the number of lags required per dataset
Send a get request at ``http://localhost:5000/lags_num?dataset_id=DATASET_NUM``, where DATASET_NUM is the number of the dataset as given in the train_splits.



### To forecast
Send a post request at ``http://localhost:5000/predict`` with the request body as specified in the session*.

****N.B. Please ensure all lags are inserted in chronological order (i.e. oldest first and latest last)***

Example:
```json
{
    "dataset_id": 118,
    "inputs_needed": 6,
    "values":[
        {"time":"2021-07-01 01:20:00","value":0.7932339991669333},
        {"time":"2021-07-01 01:30:00","value":0.5672706876771296},
        {"time":"2021-07-01 01:40:00","value":0.7102781376023531},
        {"time":"2021-07-01 01:50:00","value":0.6134815202722862},
        {"time":"2021-07-01 02:00:00","value":0.531755723360477},
        {"time":"2021-07-01 02:10:00","value":0.6511274715343153}
    ]
}
```
In this example, we forecast for dataset 118. This dataset requires 6 lags, and their values are provided as above.
