from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import glob
import os
from prophet import Prophet
from statsmodels.tsa.stattools import pacf, adfuller
import matplotlib.pyplot as plt
import re
from prophet.serialize import model_to_json
from joblib import dump
from tqdm import tqdm
import logging

# Set the logging level for cmdstanpy to a higher value (WARNING or higher)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

train_path = 'train_splits'
prophet_models_path = 'prophet_models'
lr_models_path = 'LR_models'

csv_files = glob.glob(os.path.join(train_path, '*.csv'))

metadata_df = pd.DataFrame(columns=['series_num', 'lags', 'series_type', 'test_MSE', 'time_delta'])

error_list = []


def custom_moving_average(row, lags):
    current_index = df.index.get_loc(row.name)
    start_index = max(current_index - lags, 0)  # Ensure the starting index is not negative
    subset = df.iloc[start_index:current_index]  # Get the subset of rows for the moving average
    return subset['y'].mean()


for file in tqdm(csv_files):

    series_number = re.search(r'\d+', file).group()
    try:
        df = pd.read_csv(file)
        if series_number in ['256', '439', '507']:
            df = df.drop(df.index[0])
        if 'anomaly' in df.columns:
            df = df.drop(columns=['anomaly'])
        df.columns = ['y', 'ds']
        df['ds'] = pd.to_datetime(df['ds'])
        time_diff = str(df['ds'].iloc[1] - df['ds'].iloc[0])
        df['y'] = df['y'].fillna(method='ffill')
        pacf_values = pacf(df['y'])
        pacf_threshold = 0.1
        lags = np.max(np.where(pacf_values > pacf_threshold))
        lags = 3 if lags < 3 else lags
        result = adfuller(df['y'])

        if result[1] < 0.05:
            series_type = 'additive'
        else:
            series_type = 'multiplicative'

        model = Prophet(seasonality_mode=series_type)
        model.fit(df)

        with open(os.path.join(prophet_models_path, f'model_{str(series_number)}.json'), 'w') as fout:
            fout.write(model_to_json(model))

        forecast = model.predict(df)
        # features = ["ds",
        #             "trend",
        #             "yhat_lower",
        #             "yhat_upper",
        #             "trend_lower",
        #             "trend_upper",
        #             "additive_terms",
        #             "additive_terms_lower",
        #             "additive_terms_upper",
        #             "daily",
        #             "daily_lower",
        #             "daily_upper",
        #             "weekly",
        #             "weekly_lower",
        #             "weekly_upper",
        #             "multiplicative_terms",
        #             "multiplicative_terms_lower",
        #             "multiplicative_terms_upper"]

        features = [x for x in forecast.drop(columns=['yhat'])]
        features_df = forecast[features].copy(deep=True)
        features_df['year'] = features_df['ds'].dt.year
        features_df['month'] = features_df['ds'].dt.month
        features_df['day'] = features_df['ds'].dt.day
        features_df['hour'] = features_df['ds'].dt.hour
        features_df['minute'] = features_df['ds'].dt.minute
        features_df['DoY'] = features_df['ds'].dt.day_of_year
        features_df['DoW'] = features_df['ds'].dt.day_of_week
        features_df['y'] = df['y']

        for column in features_df.columns:
            if column in ['y']:
                for i in range(1, lags + 1):
                    lagged_column_name = f"{column}_lag{i}"
                    features_df[lagged_column_name] = features_df[column].shift(i)

        if lags != 0:
            features_df['moving_average'] = df.apply(custom_moving_average, axis=1, args=(lags,))
        else:
            features_df['moving_average'] = np.zeros_like(features_df.shape[0])

        features_df = features_df.dropna()

        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(features_df.drop(columns=['ds', 'y'])[lags:],
                                                            features_df['y'][lags:],
                                                            test_size=0.3, shuffle=False)

        lr_model = RandomForestRegressor(n_estimators=25)
        lr_model.fit(X_train, y_train)

        dump(lr_model, os.path.join(lr_models_path, f'model_{str(series_number)}.joblib'), compress=7)
        preds = lr_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"\nMSE: {mse}")

        model_metadata = [series_number, lags, series_type, mse, time_diff]

        # Directly assign a new row to the DataFrame
        metadata_df.loc[len(metadata_df)] = model_metadata
    except Exception as e:
        error_list.append(series_number)
        print(e)

metadata_df.to_csv("metadata.csv", index=False)
print(error_list)
