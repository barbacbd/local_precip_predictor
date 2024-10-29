
from datetime import datetime
from collections import defaultdict
import requests
import requests_cache
from retry_requests import retry
from urllib.request import urlopen

from bs4 import BeautifulSoup, Comment
import openmeteo_requests
import pandas as pd


class State:

    def __init__(
            self,
            dataframe=None,
            exported_to_file=False,
            filename=None,
    ):
        self.df = dataframe
        self._exported_to_file = exported_to_file
        self._filename = filename if exported_to_file else None
    
    @property
    def filename(self):
        return self._filename
    
    @property
    def exported_to_file(self):
        return self._exported_to_file

    @filename.setter
    def filename(self, filename):
        self._exported_to_file = filename is not None
        self._filename = filename


def get_nao_data(
    export_to_file=False
):
    '''
    Get the North Atlantic Oscillation values for each month of the year. The 
    data starts in 1950. 

    Unlike the El Nino-Southern Oscillation, the North Atlantic Oscillation is not 
    currently predictable more than a couple weeks in advance.

    :param export_to_file: When true, the dataframe will be saved to a file nao.csv.
    :return: State object
    '''
    # Get the data from the NOAA ascii table. This table is updated monthly.
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
    html = requests.get(url).content
    # format the code as a string rather than bytes
    data_str = html.decode('utf-8')
    # Split the data by lines in the table. The first row will be the
    # header (the months of the year). The rest of the lines are the rows
    # in the table, but the first element is the year. Cut the year out
    # and set that as the row index for the dataframe.
    idxs = []
    data = []
    for line in data_str.split("\n")[1:]:
        if not line:
            continue
        split_line = line.split()
        year = split_line[0]
        split_line = split_line[1:]

        idxs.extend([
            f"{year}-{i+1}" for i in range(len(split_line))
        ])
        data.extend(split_line)

    df = pd.DataFrame(data, columns=['NAO'])
    df.index = idxs

    if export_to_file:
        df.to_csv("nao.csv")

    state = State(
        dataframe=df,
        exported_to_file=export_to_file,
        filename="nao.csv"
    )
    return state


def get_daily_data(
        latitude=36.8506,
        longitude=-75.9779,
        timezone="America/New_York",
        start_date="1950-01-01",
        end_date="1950-01-01",
        export_to_file=False
):
    '''
    Get the daily data from the api. The information included in the output:
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
    
    :param latitude: Latitude where measurements take place. [default = Latitude of Virginia Beach]
    :param longitude: Longitude where measurements take place. [default = Longitude of Virginia Beach]
    :param timezone: Timezone where the measurements take place.
    :param start_date: yyyy-mm-dd Formatted date to begin measurements.
    :param end_date: yyyy-mm-dd Formatted date to end measurements
    :param export_to_file: When true, export to a csv file with the dates as the name.

    ** Disclaimer: The majority of this function is generated by open meteo through their website. **

    :return: State object
    '''
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
	"latitude": latitude,
	"longitude": longitude,
	"start_date": start_date,
	"end_date": end_date,
	"daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "apparent_temperature_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum"
        ],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch",
	"timezone": timezone
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()
    daily_apparent_temperature_max = daily.Variables(3).ValuesAsNumpy()
    daily_apparent_temperature_min = daily.Variables(4).ValuesAsNumpy()
    daily_apparent_temperature_mean = daily.Variables(5).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(6).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(7).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(8).ValuesAsNumpy()
    
    daily_data = {
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )
    }
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
    daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
    daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["snowfall_sum"] = daily_snowfall_sum

    daily_dataframe = pd.DataFrame(data=daily_data)
    if export_to_file:
        daily_dataframe.to_csv(f'open_meteo_{start_date}_{end_date}.csv', index=False)

    state = State(
        dataframe=daily_dataframe,
        exported_to_file=export_to_file,
        filename=f'open_meteo_{start_date}_{end_date}.csv'
    )
    return state


def _is_comment(elem):
    return isinstance(elem, Comment)


def get_enso_data(export_to_file=False):
    '''
    Pull the ENSO data from NOAA's Climate Prediction Center. 

    Negative values indicate La Nina (cold)
    Positive values indicate El Nino (warm)

    The break down of values is categorized further by the following criteria:
    Weak: 0.5 - 0.9
    Moderate: 1.0 - 1.4
    Strong: 1.5 - 1.9
    Very Strong: >= 2.0

    Any value under the weak threshold is not considered strong enough to make a case
    for El Nino or La Nina categorization.

    :param start_date: yyyy-mm-dd Formatted date to begin measurements.
    :param end_date: yyyy-mm-dd Formatted date to end measurements
    :param export_to_file: When true, export to a csv file with the dates as the name.
    :return: State object
    '''
    url = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
    open_url = urlopen(url)
    soup = BeautifulSoup(open_url.read(), features="lxml")

    table_data = []
    # Find the table that contains the rolling 3 month ENSO values
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            comments = tr.find_all(string=_is_comment)
            # This is a bit of a hack, but let's find the unofficial start to the tables
            # in the webpage. 
            if "Headings" in [x.string.strip() for x in comments]:
                for td in tr.find_all("td"):
                    table_data.append(td.text.strip())
    
                break  # the main table should have all we need

        if table_data:
            break  # as long as the table data isn't empty we should have what we need

    try:
        idx = table_data.index('Year')
        table_data = table_data[idx:]

        # The data header comes in with the following values:
        # 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ'
        # The majority of the other data is setup as monthly averages. ENSO data is a rolling
        # 3 month average. Each of the 3 month periods will be given a number 1-12 to correlate
        # with the other data sets. The number will represent the middle of the 3 months. 
        # For example: DJF -> December, January, February will be given a 1, as the middle month is 
        # January.
        _header = ['Year', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        rows = []
        idxs = []
        for data in range(0, len(table_data), len(_header)):
            row = table_data[data:data+len(_header)]

            # eliminate the rows that contain the table header data
            if row != _header:
                year = row[0]
                rows.extend(row[1:])
                idxs.extend([f"{year}-{i+1}" for i in range(len(row[1:]))])

        df = pd.DataFrame(rows, columns=["ENSO"])
        df.index = idxs

        if export_to_file:
            df.to_csv("enso.csv", index=False)

        state = State(
            dataframe=df,
            exported_to_file=export_to_file,
            filename="enso.csv"
        )
        return state
            
    except ValueError:
        # failed to find the item, this is a problem
        return


def get_amo_data(export_to_file=False):
    '''
    The Atlantic Multidecadal Oscillation (AMO) is the theorized variability of the sea
    surface temperature (SST) of the North Atlantic Ocean on the timescale of several decades.

    ** Note: this is a controversial data source, and there are theories that this 
    may have little to no influence on climate patterns. **

    :param export_to_file: When true, export to a csv file with the dates as the name.
    :return: State object
    '''
    url = "https://psl.noaa.gov/data/correlation/amon.us.data"
    html = requests.get(url).content
    # format the code as a string rather than bytes
    data_str = html.decode('utf-8')

    lines = data_str.split("\n")
    # The first line of the data is the years that are included.
    # We only want data that is from the 1950+, so let's remove 
    # that date dynamically and figure out where our data ends.
    lines = [x.strip() for x in lines if x]
    start_year, end_year = lines[0].split()

    lines_to_remove = 1
    if int(start_year) < 1950:
        lines_to_remove += (1950 - int(start_year))
    
    # strip off the first n rows until we begin with 1950
    lines = lines[lines_to_remove:]

    # strip off the last n rows until we end with `end_year`
    final_data_lines = []
    for line in lines:
        final_data_lines.append(line)
        if line.startswith(end_year):
            break

    idxs = []
    data = []
    for line in final_data_lines:
        split_line = line.split()
        year = split_line[0]
        split_line = split_line[1:]

        idxs.extend([
            f"{year}-{i+1}" for i in range(len(split_line))
        ])
        data.extend(split_line)
    
    df = pd.DataFrame(data, columns=['AMO'])
    df.index = idxs

    if export_to_file:
        df.to_csv("amo.csv")

    state = State(
        dataframe=df,
        exported_to_file=export_to_file,
        filename="amo.csv"
    )
    return state

def get_ao_data(export_to_file):
    '''
    The Arctic Oscillation (AO) is a back-and-forth shifting of atmospheric pressure between the Arctic and
    the mid-latitudes of the North Pacific and North Atlantic. When the AO is strongly positive, a strong
    mid-latitude jet stream steers storms northward, reducing cold air outbreaks in the mid-latitudes. When
    the AO is strongly negative, a weaker, meandering jet dips farther south, allowing Arctic air to spill
    into the mid-latitudes.

    :param export_to_file: When true, export to a csv file with the dates as the name.
    :return: State object
    '''
    # Get the data from the NOAA ascii table. This table is updated monthly.
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii"
    html = requests.get(url).content
    # format the code as a string rather than bytes
    data_str = html.decode('utf-8')
    # Split the data by lines in the table. The first row will be the
    # header (the months of the year). The rest of the lines are the rows
    # in the table, but the first element is the year. Cut the year out
    # and set that as the row index for the dataframe.
    lines = [x for x in data_str.split("\n") if x]
    idxs = []
    monthly_data = []

    for line in lines:
        split_line = line.split()
        idxs.append(f"{split_line[0]}-{split_line[1]}")
        monthly_data.append(split_line[2])

    header = ['AO']
    df = pd.DataFrame(monthly_data, columns=header)
    df.index = idxs

    if export_to_file:
        df.to_csv("ao.csv")

    state = State(
        dataframe=df,
        exported_to_file=export_to_file,
        filename="ao.csv"
    )
    return state