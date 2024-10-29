
from statistics import mean
import datetime
import pandas as pd


_month_by_number = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}


def _parse_date(date_str):
    '''
    Parse the date out of the string in the format %Y-%M-%D HH:mm:ss
    '''
    date_split = date_str.split()[0]
    year, month, day = date_split.split("-")
    return year, month, day


def parse_daily_values_by_month(daily_value_df):
    '''
    Parse the monthly averages for each year. 

    - temperature_2m_max
    - temperature_2m_min
    - temperature_2m_mean
    - apparent_temperature_max
    - apparent_temperature_min
    - apparent_temperature_mean
    - precipitation_sum
    - rain_sum
    - snowfall_sum

    The following units are used:
    - temperature_unit: fahrenheit
    - wind_speed_unit: mph
    - precipitation_unit: inch

    :param daily_value_df: Dataframe containing the weather data for a period of time.
    :return: dictionary year -> month -> average values
    '''
    data_by_month = {}
    for index, row in daily_value_df.iterrows():

        year, month, day = _parse_date(str(row.iloc[0]))
        if year not in data_by_month:
            data_by_month[year] = {}
        if month not in data_by_month[year]:
            data_by_month[year][month] = {
                # "temperature_2m_max": [],
                # "temperature_2m_min": [],
                # "temperature_2m_mean": [],
                # "apparent_temperature_max": [],
                # "apparent_temperature_min": [],
                # "apparent_temperature_mean": [],
                "precipitation_sum": [],
                # "rain_sum": [],
                # "snowfall_sum": [],
            }
        
        # if row.iloc[1] is not None:
        #     data_by_month[year][month]["temperature_2m_max"].append(row.iloc[1])
        # if row.iloc[2] is not None:
        #     data_by_month[year][month]["temperature_2m_min"].append(row.iloc[2])
        # if row.iloc[3] is not None:
        #     data_by_month[year][month]["temperature_2m_mean"].append(row.iloc[3])
        # if row.iloc[4] is not None:
        #     data_by_month[year][month]["apparent_temperature_max"].append(row.iloc[4])
        # if row.iloc[5] is not None:
        #     data_by_month[year][month]["apparent_temperature_min"].append(row.iloc[5])
        # if row.iloc[6] is not None:
        #     data_by_month[year][month]["apparent_temperature_mean"].append(row.iloc[6])
        if row.iloc[7] is not None:
            data_by_month[year][month]["precipitation_sum"].append(row.iloc[7])
        # if row.iloc[8] is not None:
        #     data_by_month[year][month]["rain_sum"].append(row.iloc[8])
        # if row.iloc[9] is not None:
        #     data_by_month[year][month]["snowfall_sum"].append(row.iloc[9])

    averages = {}
    for year, yearly_value in data_by_month.items():
        for month, monthly_value in yearly_value.items():
            index = f"{year}-{int(month)}"

            for key in monthly_value:
               averages[index] = mean(monthly_value[key])
    
    df = pd.DataFrame.from_dict(averages, orient='index', columns=["precipitation_sum"])

    return df


def combine_data(dfs):
    '''
    Combine the dataframes. Each dataframe should be cut down to be the
    same size as the smallest dataframe to avoid issues/missing values.
    '''
    smallest_df_len = min([len(df) for df in dfs])
    combined_df = dfs[0].head(smallest_df_len)

    for df in dfs[1:]:
        sliced_df = df.head(smallest_df_len)
        combined_df = pd.concat([combined_df, sliced_df], axis=1, join='outer')
    
    return combined_df