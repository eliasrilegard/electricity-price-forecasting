"""
COPY F19
N-BEATS with covariates(load+datetime)
Better version of F17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler    # Speed up the training of our model
from darts.utils.callbacks import TFMProgressBar

# def generate_torch_kwargs():
#     # run torch models on CPU, and disable progress bars for all model stages except training.
#     return {
#         "pl_trainer_kwargs": {
#             "accelerator": "cpu",
#             "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
#         }
#     }


# Part1:Reading dataset price
df_prices = pd.read_csv("Day-ahead Prices_202301010000-202401010000.csv")
df_prices["MTU (CET/CEST)"] = pd.to_datetime(df_prices["MTU (CET/CEST)"].str.split(" - ").str[0], format="%d.%m.%Y %H:%M")
df_prices.set_index("MTU (CET/CEST)", inplace = True)
df_prices.drop(columns = [f'BZN|SE4', 'Currency'], inplace = True)
df_prices = df_prices.ffill()
df_prices = df_prices[~df_prices.index.duplicated(keep ="first")]
# print(df_prices)

series_DayaheadPrices = TimeSeries.from_dataframe(df_prices)    #Since we are working with Darts, we will go from DataFrame to a TimeSeries object, which is the fundamental object in Darts. Every model in Darts must have a TimeSeries object as input, and it outputs a TimeSeries object as well.
# series_DayaheadPrices.plot(label="Day-ahead Price")
# plt.show()


# Part1:Reading dataset load
df_load = pd.read_csv("Total Load - Day Ahead _ Actual_202301010000-202401010000.csv")
df_load['Time (CET/CEST)'] = pd.to_datetime(df_load['Time (CET/CEST)'].str.split(' - ').str[0],format='%d.%m.%Y %H:%M')
df_load.set_index("Time (CET/CEST)", inplace = True)
df_load.drop(columns = ['Actual Total Load [MW] - BZN|SE4'], inplace = True)
df_load = df_load.ffill()
df_load = df_load[~df_load.index.duplicated(keep ="first")]
# print(df_load)

series_DayaheadLoad = TimeSeries.from_dataframe(df_load)
# series_DayaheadLoad.plot(label="Day-ahead Total Load Forecast [MW]")
# plt.show()


# Part2:Create training and validation sets:
train, test = series_DayaheadPrices[:-120], series_DayaheadPrices[-120:]        # Last 5 days


# Part2: normalized train(price)
train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)    # Fit the scaler on the training set only, because the model is not supposed to have information coming from the test set.
# scaled_train.plot(label='Scaled_train (untill 27 Dec 2023)')
# plt.show()


# Part2: normalized load
predicted_load_series = Scaler()
scaled_load = predicted_load_series.fit_transform(series_DayaheadLoad)
# scaled_load.plot(label='Scaled_load')
# plt.show()

"""
# predicted_load_series = TimeSeries.from_dataframe(df_load)

# predicted_load_series = Scaler().fit_transform(series_DayaheadLoad.map(lambda ts, x: np.log(x)))
# plt.figure(figsize=(12, 5))
# predicted_load_series.plot()
# plt.show()
"""

from darts.utils.timeseries_generation import datetime_attribute_timeseries

month_series = datetime_attribute_timeseries(scaled_load, attribute="month")
month_series = Scaler().fit_transform(month_series)


weekday_series = datetime_attribute_timeseries(month_series, attribute="weekday")
weekday_series = Scaler().fit_transform(weekday_series)


hour_series = datetime_attribute_timeseries(weekday_series, attribute="hour")
hour_series = Scaler().fit_transform(hour_series)



datetime_series = month_series.stack(weekday_series).stack(hour_series)

# plt.figure(figsize=(12, 5))
# datetime_series.plot()
# # plt.xlim(
# #   left=datetime_series.start_time() - pd.DateOffset(days=6),
# #   right=datetime_series.start_time() + pd.DateOffset(months=1, days=10)
# # )
# plt.show()


# load + datetime cov like: month, hour and weekday
covariates = scaled_load.stack(datetime_series)
# covariates.plot()
# plt.show()


from darts.models import NBEATSModel

nbeats_cov = NBEATSModel(
    input_chunk_length=168,
    output_chunk_length=24,
    generic_architecture=True,
    num_stacks=5,
    num_blocks=1,
    num_layers=4,
    layer_widths=600,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=90,
    random_state=42,
    # dropout=0.1,
    # **generate_torch_kwargs(),
)


nbeats_cov.fit(
    scaled_train,
    past_covariates=covariates,
    #epochs=10   # why?
)


scaled_pred_nbeats_cov = nbeats_cov.predict(past_covariates=covariates, n=120)
pred_nbeats_cov = train_scaler.inverse_transform(scaled_pred_nbeats_cov)    # Of course, the predictions are scaled as well, so we need to reverse the transformation.

test.plot(label='test')
pred_nbeats_cov.plot(label='N-BEATS')
plt.show()


from darts.metrics import mae
mae_nbeats = mae(test, pred_nbeats_cov)
print(mae_nbeats)


from darts.metrics import mape
print("MAPE: {:.2f}%.".format(mape(test, pred_nbeats_cov)))