## Guide

### To run the server

run "docker-compose up"

### To enquire about the number of lags required per dataset
Send a get request at "http://localhost:5000/lags_num?dataset_id={DATASET_NUM}", where DATASET_NUM is the number of the dataset as given in the train_splits.

### To forecast
Send a post request at "http://localhost:5000/predict" with the request body as specified in the session*.

****N.B. Please ensure all lags are inserted in chronological order (i.e. oldest first and latest last)***