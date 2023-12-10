import numpy as np
from flask import Flask, request, jsonify, abort
import pandas as pd
from prophet.serialize import model_from_json
from joblib import load

app = Flask(__name__)


# Your prediction model import and setup
# Replace with your model import
# Example:
# from my_prediction_model import predict_future_value

@app.route('/lags_num', methods=['GET'])
def get_lags_num():
    num = int(request.args.get('dataset_id'))

    meta_data = pd.read_csv('metadata.csv')
    lags_number = list(meta_data[meta_data['series_num'] == num]['lags'])[0]
    return jsonify({'inputs_needed': lags_number})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from the POST request
        dataset_id = data.get('dataset_id')
        metadata = pd.read_csv('metadata.csv')
        lags_number = list(metadata[metadata['series_num'] == dataset_id]['lags'])[0]

        values = []
        if lags_number != 0:
            values = data.get('values', [])
            if len(values) != lags_number:
                abort(404, description=f"Invalid number of lags provided. Please provide {lags_number} values")

        prediction_df = pd.DataFrame([pd.to_datetime(values[-1]['time']) +
                                      pd.to_timedelta(list(
                                          metadata[metadata['series_num'] == dataset_id]['time_delta'])[
                                                          0])])
        prediction_df.columns = ['ds']
    except Exception as e:
        print(e)
        abort(404, description=f"Invalid JSON body.")

    with open(f'prophet_models/model_{dataset_id}.json', 'r') as fin:
        prophet = model_from_json(fin.read())

    forecast = prophet.predict(prediction_df)
    forecast['year'] = forecast['ds'].dt.year
    forecast['month'] = forecast['ds'].dt.month
    forecast['day'] = forecast['ds'].dt.day
    forecast['hour'] = forecast['ds'].dt.hour
    forecast['minute'] = forecast['ds'].dt.minute
    forecast['DoY'] = forecast['ds'].dt.day_of_year
    forecast['DoW'] = forecast['ds'].dt.day_of_week
    forecast = forecast.drop(columns=['ds', 'yhat'])

    reversed_values = values[::-1]
    for idx, lag in enumerate(reversed_values):
        forecast[f'y_lag{idx + 1}'] = None
        forecast.loc[0, f'y_lag{idx + 1}'] = lag['value']

    forecast = forecast.fillna(method='ffill')

    if lags_number == 0:
        forecast['moving_average'] = 0
    else:
        forecast['moving_average'] = None
        values_list = [d['value'] for d in values]

        forecast.loc[0, 'moving_average'] = sum(values_list)/len(values_list)

    # column_order = ['trend','yhat_lower','yhat_upper','trend_lower','trend_upper','additive_terms','additive_terms_lower','additive_terms_upper','daily','daily_lower','daily_upper','weekly','weekly_lower','weekly_upper','multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper','year','month','day','hour','minute','DoY','DoW']
    # for index in range(lags_number):
    #     column_order.append(f'y_lag{index+1}')
    # column_order.append('moving_average')
    forecast = forecast.reindex(sorted(forecast.columns), axis=1)
    # Define the new order of columns
    # forecast.to_csv('test_api.csv', index=False)

    lr_model = load(f"LR_models/model_{dataset_id}.joblib")
    prediction = lr_model.predict(forecast)[0]


    # Return the predicted value as JSON
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
