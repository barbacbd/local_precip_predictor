from datetime import datetime
from .web import get_daily_data, get_nao_data, get_enso_data
from .analysis import parse_daily_values_by_month
from json import dumps

import pandas as pd

def main():
    today = datetime.now()
    end_date = f"{today.year}-{today.month:02}-{today.day:02}"

    export_to_file = True

    # daily_data_state = get_daily_data(end_date=end_date, export_to_file=export_to_file)
    # nao_state = get_nao_data(export_to_file=export_to_file)
    # enso_state = get_enso_data(export_to_file=export_to_file)

    # Temp, This is only being used to drive the number of API requests down.
    # The open meteo api has an hourly limit.
    df = pd.read_csv('open_meteo_1950-01-01_2024-10-28.csv')

    averages = parse_daily_values_by_month(daily_value_df=df)
    print(dumps(averages["2023"], indent=2))

if __name__ == '__main__':
    main()