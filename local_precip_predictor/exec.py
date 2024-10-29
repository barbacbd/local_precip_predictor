from datetime import datetime
from .web import get_daily_data, get_nao_data, get_enso_data, get_amo_data, get_ao_data
from .analysis import parse_daily_values_by_month, combine_data
from json import dumps

import pandas as pd

def main():
    today = datetime.now()
    end_date = f"{today.year}-{today.month:02}-{today.day:02}"

    export_to_file = False

    # daily_data_state = get_daily_data(end_date=end_date, export_to_file=export_to_file)
    nao_state = get_nao_data(export_to_file=export_to_file)
    enso_state = get_enso_data(export_to_file=export_to_file)

    # Temp, This is only being used to drive the number of API requests down.
    # The open meteo api has an hourly limit.
    df = pd.read_csv('open_meteo_1950-01-01_2024-10-28.csv')

    averages = parse_daily_values_by_month(daily_value_df=df)

    amo_state = get_amo_data(export_to_file=export_to_file)
    ao_state = get_ao_data(export_to_file=export_to_file)

    df = combine_data([nao_state.df, enso_state.df, amo_state.df, ao_state.df, averages])
    df.to_csv('test.csv')


if __name__ == '__main__':
    main()