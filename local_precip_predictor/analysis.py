
from statistics import mean
import datetime


def _parse_date(date_str):
    '''
    Parse the date out of the 
    '''
    date_split = date_str.split()[0]
    year, month, day = date_split.split("-")
    return year, month, day


def parse_daily_values_by_month(daily_value_df):
    '''
    Fill in here
    '''
    data_by_month = {}
    for index, row in daily_value_df.iterrows():

        year, month, day = _parse_date(str(row[0]))
        print(f"Year = {year}, Month = {month}, Day = {day}")
        if year not in data_by_month:
            data_by_month[year] = {}
        if month not in data_by_month[year]:
            data_by_month[year][month] = {
                "temperature_2m_max": [],
                "temperature_2m_min": [],
                "temperature_2m_mean": [],
                "apparent_temperature_max": [],
                "apparent_temperature_min": [],
                "apparent_temperature_mean": [],
                "precipitation_sum": [],
                "rain_sum": [],
                "snowfall_sum": [],
            }
        
        if row[1] is not None:
            data_by_month[year][month]["temperature_2m_max"].append(row[1])
        if row[2] is not None:
            data_by_month[year][month]["temperature_2m_min"].append(row[2])
        if row[3] is not None:
            data_by_month[year][month]["temperature_2m_mean"].append(row[3])
        if row[4] is not None:
            data_by_month[year][month]["apparent_temperature_max"].append(row[4])
        if row[5] is not None:
            data_by_month[year][month]["apparent_temperature_min"].append(row[5])
        if row[6] is not None:
            data_by_month[year][month]["apparent_temperature_mean"].append(row[6])
        if row[7] is not None:
            data_by_month[year][month]["precipitation_sum"].append(row[7])
        if row[8] is not None:
            data_by_month[year][month]["rain_sum"].append(row[8])
        if row[9] is not None:
            data_by_month[year][month]["snowfall_sum"].append(row[9])

    averages = {}
    for year, yearly_value in data_by_month.items():
        if year not in averages:
            averages[year] = {}

        for month, monthly_value in yearly_value.items():
            if month not in averages[year]:
                averages[year][month] = {}

            for key in monthly_value:
                averages[year][month][key] = mean(monthly_value[key])
    
    return averages
