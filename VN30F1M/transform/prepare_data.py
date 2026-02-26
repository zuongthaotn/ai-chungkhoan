from VN30F1M.transform import OHLCV_DIR, DATA_READY_DIR
import os
import pandas as pd
from VN30F1M.transform.func_validating import validate_ohlcv_dataset


if __name__ == '__main__':
    ohlcv_file = str(OHLCV_DIR) + '/VN30F1M_5minutes.csv'
    csv_ready_file = str(DATA_READY_DIR) + '/VN30F1M_5minutes_ready.csv'
    is_file = os.path.isfile(ohlcv_file)
    if is_file:
        ohlcv_data = pd.read_csv(ohlcv_file, index_col='Date', parse_dates=True)
        validated_data = validate_ohlcv_dataset(ohlcv_data)
        print(validated_data)
    else:
        print(f"File {ohlcv_file} not found.")