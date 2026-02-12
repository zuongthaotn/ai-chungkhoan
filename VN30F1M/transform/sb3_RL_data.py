import os
from pathlib import Path
import numpy as np
import pandas as pd
import numpy as np
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')


def get_open_price_position(r):
    if r["Open"] > r["prev_Close"]:
        return -0.5
    if r["Open"] == r["prev_Close"]:
        return 0
    if r["Open"] < r["prev_Close"]:
        return 0.5


def get_ibs_vol_group(r):
    if r["Volume"] > r["prev_Vol"] and r["ibs"] > r["prev_ibs"]:
        return -1
    if r["Volume"] > r["prev_Vol"] and r["ibs"] < r["prev_ibs"]:
        return -0.5
    if r["Volume"] < r["prev_Vol"] and r["ibs"] > r["prev_ibs"]:
        return 0.5
    if r["Volume"] < r["prev_Vol"] and r["ibs"] < r["prev_ibs"]:
        return 1


def get_close_price_position(r):
    if r["Close"] > r["prev_High"]:
        return -1
    if r["Close"] > max(r["prev_Close"], r["prev_Open"]):
        return -0.5
    if max(r["prev_Close"], r["prev_Open"]) > r["Close"] > min(r["prev_Close"], r["prev_Open"]):
        return 0
    if r["Close"] < min(r["prev_Close"], r["prev_Open"]):
        return 0.5
    if r["Close"] < r["prev_Low"]:
        return 1


def labeling_data(df):
    tmp_data = df.copy()
    tmp_data['DayHigh'] = tmp_data['High']
    daily_data = tmp_data.resample('D').agg({
            'DayHigh': 'max'
        })
    daily_data.dropna(subset=['DayHigh'], inplace=True)
    #
    data = df.copy()

    data["ema_fast"] = ta.ema(data["Close"], length=20)
    data["ema_low"] = ta.ema(data["Close"], length=250)
    data["ema_cross"] = ((data["ema_fast"] > data["ema_low"]) & (data["ema_fast"].shift(1) <= data["ema_low"].shift(1)) | (data["ema_fast"] < data["ema_low"]) & (data["ema_fast"].shift(1) >= data["ema_low"].shift(1)))
    data = data.assign(time_d=pd.PeriodIndex(data.index, freq='1D').to_timestamp())
    #
    merged_data = pd.merge(data, daily_data, left_on="time_d", right_index=True, how="left")
    merged_data['hour'] = merged_data.index.hour
    merged_data['minute'] = merged_data.index.minute
    merged_data['prev_High'] = merged_data['High'].shift(1)
    merged_data['prev_Low'] = merged_data['Low'].shift(1)
    merged_data['prev_Close'] = merged_data['Close'].shift(1)
    merged_data['prev_Open'] = merged_data['Open'].shift(1)
    merged_data['prev_Vol'] = merged_data['Volume'].shift(1)
    merged_data['is_max'] = merged_data.apply(lambda r: 1 if r["High"] == r["DayHigh"] else 0, axis=1)
    ana_data = merged_data.dropna()
    ana_data['upper_shadow'] = ana_data.apply(lambda r: r["High"] - max(r["Open"], r["Close"]), axis=1)
    ana_data['prev_upper_shadow'] = ana_data['upper_shadow'].shift(1)
    ana_data['ibs'] = ana_data.apply(
        lambda r: 0 if r["High"] == r["Low"] else (r["Close"] - r["Low"]) / (r["High"] - r["Low"]), axis=1)
    ana_data['prev_ibs'] = ana_data['ibs'].shift(1)
    ana_data['RSI20'] = ta.rsi(ana_data["Close"], length=20)
    ana_data['RSI10'] = ta.rsi(ana_data["Close"], length=10)
    ana_data['avg_Volume'] = ana_data['Volume'].rolling(20).mean()
    ana_data['prev_avg_Volume'] = ana_data['avg_Volume'].shift(1)
    ana_data["MB"] = ana_data["Close"].rolling(20).mean()
    ana_data["STD"] = ana_data["Close"].rolling(20).std()
    ana_data["UB"] = ana_data["MB"] + 1.5 * ana_data["STD"]
    #
    ana_data['upper_wick_group'] = ana_data.apply(
        lambda r: 1 if r["upper_shadow"] > r["prev_upper_shadow"] else -1, axis=1)
    ana_data["ibs_vol_group"] = ana_data.apply(lambda r: get_ibs_vol_group(r), axis=1)
    ana_data['rsi_area'] = ana_data.apply(
        lambda r: 1 if r["RSI20"] > 55 else (0.33 if r["RSI20"] < 45 else 0.66), axis=1)
    ana_data['higher_high_lower_vol'] = ana_data.apply(
        lambda r: 1 if (r["High"] > r["prev_High"] and r["Volume"] < r["prev_Vol"]) else -1, axis=1)
    ana_data['Volume_higher_avg'] = ana_data.apply(lambda r: 1 if r["Volume"] > r["avg_Volume"] else -1, axis=1)
    ana_data['Volume_vs_prev_Vol'] = ana_data.apply(lambda r: 1 if r["Volume"] > r["prev_Vol"] else -1, axis=1)
    ana_data['Volume_avg_group'] = ana_data.apply(
        lambda r: 1 if r["avg_Volume"] > r["prev_avg_Volume"] else -1, axis=1)
    ana_data['close_price_group'] = ana_data.apply(lambda r: get_close_price_position(r), axis=1)
    ana_data['open_price_group'] = ana_data.apply(lambda r: get_open_price_position(r), axis=1)
    ana_data['High_position'] = ana_data.apply(lambda r: 1 if r["High"] > r["UB"] else -1, axis=1)
    ana_data["BB_rejection"] = ana_data.apply(lambda r: 1 if r["Close"] < r["UB"] else -1, axis=1)
    ana_data.dropna(inplace=True)
    return ana_data

if __name__ == '__main__':
    csv_dir = Path(__file__).parent.parent
    csv_file = str(csv_dir) + '/VN30F1M_5minutes.csv'
    saved_file = str(csv_dir) + '/transform/sb3_ready_data.csv'
    is_file = os.path.isfile(csv_file)
    if is_file:
        dataset = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        dataset = labeling_data(dataset)
        dataset.to_csv(saved_file)
