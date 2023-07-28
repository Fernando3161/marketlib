'''
Created on 20.01.2022

@author: Fernando Penaherrera @UOL/OFFIS
@author: Steffen Wehkamp OFFIS

Create market price time series from historical data
Using price pattern generated from historical data from 2015-2019

'''
from scipy.stats import norm
import warnings
from pandas.errors import SettingWithCopyWarning
import logging
from os.path import join
import pandas as pd
import copy
import datetime
import sys
import os
import numpy as np
path = os.path.realpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir))

sys.path.append(path)
from src.electricity_markets.common import ELECTRICITY_MARKETS_DIR, PROC_DATA_DIR, RAW_DATA_DIR, REAL_RAW_DATA_DIR
from src.electricity_markets.lehmer import LehmerRandom

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
logging.basicConfig(level=logging.INFO)


def dow(x):
    """A function to change the python day of the week numbering to this project
    equivalent day of the week format. 

    Args:
        x (int): The day of the week according to datetime standars. Monday is 0

    Returns:
        int: The equivalend Day of Week used for price patterns. 
    """
    conversion = {0: 2, 1: 4, 2: 4, 3: 4, 4: 6, 5: 7, 6: 1}
    return conversion[x]


def create_price_pattern(year, market, mean_val=None, normalization=True):
    '''
    Creates a Price Pattern for the Day Ahead Price.
    Uses existing profiles to compose a pattern for a whole year

    :param year: Desired year
    :param market: Day Ahead or Intra Day. One of ["da", "id"]
    :param mean_val: Mean value. Optional for the years 2015-2020, since data exists.
    '''

    if market not in ["da", "id"]:
        raise ValueError('Parameter "market" must be one "da" or "id".')

    START = f"{year}-01-01 00:00:00"

    if market == "da":
        END = f"{year}-12-31 23:00:00"
        dti = pd.date_range(start=START, end=END, freq="H", tz='Europe/Berlin')

    if market == "id":
        END = f"{year}-12-31 23:45:00"
        dti = pd.date_range(
            start=START,
            end=END,
            freq="15T",
            tz='Europe/Berlin')

    periods = len(dti)

    market_df = pd.DataFrame()

    # Need some parameters of the Datetime object to search patterns
    import numpy as np

    random_generator = LehmerRandom(seed=12345)

    market_df["Date"] = dti
    market_df["Month"] = dti.month
    market_df["DayOfWeek"] = dti.dayofweek
    market_df["EquivalentDayOfWeek"] = [dow(x) for x in market_df["DayOfWeek"]]
    if market == "da":
        probabilities = [random_generator.random()
                         for _ in range(4*len(market_df["Date"]))]
        market_df["Probability"] = probabilities[::4]

    if market == "id":
        market_df["Probability"] = [random_generator.random()
                                    for _ in range(len(market_df["Date"]))]
    EXCEL_DATA = join(REAL_RAW_DATA_DIR, "SPOT_PP_YearWeekly.xlsx")

    # Excel reader
    if market == "da":
        sheet = "PP_DA_YearWeek"
    elif market == "id":
        sheet = "PP_ID_YearWeek"

    price_data = pd.read_excel(
        EXCEL_DATA,
        sheet,
        engine='openpyxl',
        parse_dates=False,
        skiprows=1)

    price_data
    price_data.drop(["Raw", "END"], axis=1, inplace=True)
    price_data = price_data.head(60)

    # read volatility data
    EXCEL_DATA_VOLAT = join(REAL_RAW_DATA_DIR, "SPOT_VP_YearWeekly.xlsx")

    # Excel reader
    if market == "da":
        sheet = "VP_DA_YearWeek"
    elif market == "id":
        sheet = "VP_ID_YearWeek"

    volatility_data = pd.read_excel(
        EXCEL_DATA_VOLAT,
        sheet,
        engine='openpyxl',
        parse_dates=False,
        skiprows=1)

    volatility_data
    volatility_data.drop(["END"], axis=1, inplace=True)
    volatility_data.dropna(axis=1, how="all", inplace=True)

    volatility_data = volatility_data.head(60)

    price_info = {
        "price": price_data,
        "volatility": volatility_data
    }

    for key, price_data in price_info.items():
        i = 0
        price = []
        while i < periods:
            month = market_df.at[i, "Month"]
            # Python puts monday at 0
            day_of_week = market_df.at[i, "EquivalentDayOfWeek"]
            # Filter data to search the corresponding values
            da_data_month = price_data[price_data["Month"] == month]
            da_data_day = da_data_month[da_data_month["TypDay"] == day_of_week]
            # Easier to transpose the table
            partial_data = da_data_day.T
            cols = partial_data.columns
            partial_data = partial_data.drop(
                ["Month", "TypDay"]
            )
            if market == "da":
                price = price + list(partial_data[cols[0]][0:24])
                i += 24

            elif market == "id":
                price = price + list(partial_data[cols[0]][0:24*4])
                i += 24 * 4

        market_df[key] = price

    market_df["price_uc"] = norm.ppf(
        market_df["Probability"], loc=market_df["price"], scale=market_df["volatility"])

    market_df.head()
    market_df["market_price"] = market_df["price_uc"]

    # This normalizes the resulting data so the mean is 100
    # print("Mean before normalization: ", np.mean(market_df["market_price"]))

    # if normalization:
    #     market_df["market_price"]*= 100/np.mean(market_df["market_price"])

    # print("Mean after normalization: ", np.mean(market_df["market_price"]))

    if mean_val:
        # Override the previous one if mean value is given
        # The values of the profiles in the excel data are normalized to 100
        market_df["market_price_year"] = market_df["market_price"] * \
            mean_val / 100
    else:
        market_df["market_price_year"] = market_df["market_price"]

    if year not in range(2015, 2021) and mean_val is None:
        raise ValueError(
            "For years outside of 2015-2020 a mean value is required. I.e.: 'mean_val=40'")

    # Take only the necessary columns
    res = market_df[["Date", "market_price_year"]]

    # Reformat Dataframe to avoid conflicts with other markets
    if market == "da":
        new_column_name = f"day_ahead"
    elif market == "id":
        new_column_name = f"intra_day"
    res.rename(columns={"market_price_year": new_column_name}, inplace=True)
    res.set_index("Date", inplace=True)

    logging.info(
        "{} Price pattern for the year {} created".format(
            market.upper(), year))

    # Uncomment to write results
    # market_df.to_csv(f"market_{year}_{market}_mean_{mean_val}.csv")

    return res


def create_markets_info(
        year,
        mean_da=None,
        mean_id=None,
        fb=None,
        fp=None,
        save_csv=True,
        use_real_data=False):
    '''
    Creates a dataframe with information on the IntraDay, Day Ahead, Future Base, and Future Peak
    markets

    For years 2015-2017: Uses DA and ID market data, FP and FB must be given
    For years 2018-2020: Uses DA, ID, FP, and FB market data. None must be given
    For years 2021-2025: Uses FB and FP market data. DA and ID must be given
    For years 2025-: DA, ID, FP and FP market data must be giiven

    :param year: Year for data
    :param mean_da: Mean Day Ahead price. Required for years 2022 an onwards
    :param mean_id: Mean Intraday price. Required for years 2022 an onwards
    :param fb: Future Base Prices. Required for years outside of 2018-2025
    :param fp: Future Peak Prices. Required for years outside of 2018-2025
    '''
    # Check the years:
    if year < 2015:
        raise ValueError("Year has to be greater than 2015")

    # if year >= 2021 and mean_da is None:
    #     raise ValueError(
    #         'Mean Day ahead price "mean_da=" must be given for year>2021')

    # if year >= 2021 and mean_id is None:
    #     raise ValueError(
    #         'Mean Intraday price "mean_id=" must be given for year>2021')

    if year not in range(2018, 2027) and fb is None:
        raise ValueError(
            'Future Base price "fb=" must be given for years not in 2018-2026')

    if year not in range(2018, 2027) and fp is None:
        raise ValueError(
            'Future Peak price "fp=" must be given for years not in 2018-2027')

    # Get Markets Info
    MARKET_DATA_FILE = join(REAL_RAW_DATA_DIR, "Market_Assumptions.xlsx")
    markets_price_data = pd.read_excel(MARKET_DATA_FILE, "Market Assumptions",
                                       index_col="Year",
                                       engine='openpyxl',
                                       skiprows=3, nrows=12

                                       )

    # Get DA and ID info
    if year in range(2015, 2027):
        mean_da = markets_price_data.at[year, "Day Ahead"]
        mean_id = markets_price_data.at[year, "Intraday"]

    day_ahead = create_price_pattern(
        year=year, market="da", mean_val=mean_da)
    day_ahead = day_ahead.resample("15min").ffill()
    intra_day = create_price_pattern(
        year=year, market="id", mean_val=mean_id)
    markets_data = pd.concat([day_ahead, intra_day], axis=1)

    # Need to copy the last 3 values to fill the table for Day Ahead
    for i in range(1, 4):
        markets_data["day_ahead"][-i] = markets_data["day_ahead"][-4]

    if year in range(2018, 2027):
        future_base = markets_price_data.at[year, "Base"]
    else:
        future_base = fb

    markets_data["future_base"] = [future_base] * markets_data.shape[0]

    if year in range(2018, 2027):
        future_peak = markets_price_data.at[year, "Peak"]
    else:
        future_peak = fp

    future_peak_vals = [future_peak] * markets_data.shape[0]

    day_of_the_week = markets_data.index.dayofweek

    for i in range(markets_data.shape[0]):
        # Make the future peaks value 0 outside 8h and 21h exclusive (up to
        # 20h45)

        if markets_data.index[i].hour not in range(8, 21):
            future_peak_vals[i] = 0

        # Make weekend values == 0
        if day_of_the_week[i] in [5, 6]:
            future_peak_vals[i] = 0

    markets_data["future_peak"] = future_peak_vals

    time_stamps = markets_data.index.tolist()
    utc_offset = [t.tz._utcoffset for t in time_stamps]

    # check those where the offset is different from the 01-Jan 00:00:00
    diff = [False if utc == utc_offset[0] else True for utc in utc_offset]

    # now a quick solution is to move the prices 1 hour up for the times with
    # utc +2
    da_utc = [markets_data["day_ahead"][i] if diff[i] ==
              False else markets_data["day_ahead"][i + 4] for i in range(0, len(diff))]
    id_utc = [markets_data["intra_day"][i] if diff[i] ==
              False else markets_data["intra_day"][i + 4] for i in range(0, len(diff))]

    markets_data["day_ahead"] = da_utc
    markets_data["intra_day"] = id_utc

    logging.info(f"Electricity market prices (DA,ID,FB,FP) for {year} created")

    # Write the dataframe to a csv
    if save_csv:
        if os.path.isdir(PROC_DATA_DIR):
            markets_data.to_csv(
                join(
                    PROC_DATA_DIR,
                    "EnergyMarketPrice_{}.csv".format(year)))

        else:
            markets_data.to_csv(
                join(
                    os.getcwd(),
                    "EnergyMarketPrice_{}.csv".format(year)))
    if not use_real_data:
        logging.info("Using profiles for market price data information")

        return markets_data

    if use_real_data:
        logging.info("Using real market price data information")
        real_market_data = copy.deepcopy(markets_data)
        da_price = get_real_values_da(year=year)
        real_market_data["day_ahead"] = [p for p in da_price]
        id_price = get_real_values_id(year=year)
        real_market_data["intra_day"]= [p for p in id_price]
        return real_market_data




def get_real_values_da(year=2018):
    # Get Path for the real data of DA
    DA_EXCEL_REAL = os.path.join(
        ELECTRICITY_MARKETS_DIR, "raw", "real_data", "DA_RawData.xlsx")
    if not os.path.isfile(DA_EXCEL_REAL):
        raise AssertionError(
            f"File with Day Ahead Data does not exist: \n {DA_EXCEL_REAL}")

    if year not in range(2015,2020):
        raise ValueError("Year must be between 2015 and 2019")


    # Read excel
    da_real_data = pd.read_excel(DA_EXCEL_REAL, "2015-2019",
                                 engine='openpyxl',
                                 skiprows=7)
    # clean the data
    da_real_data.drop(0, inplace=True)
    da_real_data = da_real_data[:-1]

    # Get only the required columns
    cols = da_real_data.columns

    da_data = da_real_data[[cols[i] for i in [2, 3, 6]]]
    da_data_temp = copy.deepcopy(da_data)
    cols_new = da_data_temp.columns
    new_names = ["Day", "Time", "Price"]
    da_data_temp.rename(
        columns={cols_new[i]: new_names[i] for i in range(3)}, inplace=True)

    da_data_temp["Date"] = np.nan
    for idx, row in da_data_temp.iterrows():
        mins = row["Time"].hour*60
        time_change = datetime.timedelta(minutes=mins)
        da_data_temp.at[idx, "Date"] = row["Day"]+time_change

    # add a row at the end
    indx = max(da_data_temp.index)

    # Add a row to the last so I can refill every 15 mins
    da_data_temp.loc[indx+1] = [np.nan,
                                np.nan,
                                da_data_temp.at[indx, "Price"],
                                da_data_temp.at[indx, "Date"]+datetime.timedelta(minutes=60)]

    # Select only price data
    da_data_final = da_data_temp[["Date", "Price"]]

    # Create a Date Time so I can pass the proper index
    idx_0 = min(da_data_final.index)
    start = da_data_final.at[idx_0, "Date"]
    dates = pd.date_range(start, periods=da_data_final.shape[0], freq="H")
    da_data_final["Date"] = dates
    da_data_final.set_index("Date", inplace=True)

    # Resample to 15 mins
    da_data_final_15m = da_data_final.resample("15min").ffill()

    # Use Queries to filter the data to the required year
    da_data_year_15m = da_data_final_15m.query(
        f"index < '{year+1}-01-01 00:00:00'")
    da_data_year_15m = da_data_year_15m.query(
        f"index >= '{year}-01-01 00:00:00'")
    da_price = da_data_year_15m["Price"]

    return da_price

def get_real_values_id(year=2018):
    ID_EXCEL_REAL = os.path.join(ELECTRICITY_MARKETS_DIR,"raw", "real_data","ID_RawData.xlsx")
    if not os.path.isfile(ID_EXCEL_REAL):
        raise AssertionError(
            f"File with Intraday Data does not exist: \n {ID_EXCEL_REAL}")
    
    if year not in range(2015,2020):
        raise ValueError("Year must be between 2015 and 2019")
    # Read excel for Intra Day
    id_real_data = pd.read_excel(ID_EXCEL_REAL, "2015-2019",
                                        # index_col="Year",
                                        engine='openpyxl',
                                        skiprows=7)
        
    id_real_data.drop(0, inplace=True)
    id_real_data=id_real_data[:-1]
    cols = id_real_data.columns


    # Get only the required columns
    id_data= id_real_data[[cols[i] for i in [2,3,6]]]
    id_data_temp = copy.deepcopy(id_data)
    cols_new = id_data_temp.columns
    new_names = ["Day", "Time", "Price"]
    id_data_temp.rename(columns = {cols_new[i]:new_names[i] for i in range(3)}, inplace=True)

    id_data_temp["Date"] = np.nan
    for idx,row in id_data_temp.iterrows():
        mins = row["Time"].hour*60 + row["Time"].minute
        time_change = datetime.timedelta(minutes= mins)
        id_data_temp.at[idx,"Date"] = row["Day"]+time_change

    #add a row at the end
    indx = max(id_data_temp.index)

    # Add a row to the last so I can refill every 15 mins
    id_data_temp.loc[indx+1] = [np.nan, 
                                np.nan,
                                id_data_temp.at[indx, "Price"],
                                id_data_temp.at[indx,"Date"]+datetime.timedelta(minutes=60)]

    # Select only price data
    id_data_final= id_data_temp[["Date", "Price"]]

    # Create a Date Time so I can pass the proper index
    idx_0 = min(id_data_final.index)
    start = id_data_final.at[idx_0,"Date"]
    dates = pd.date_range(start, periods=id_data_final.shape[0], freq="15T")
    id_data_final["Date"]=dates
    id_data_final.set_index("Date", inplace=True)

    # Resample to 15 mins
    id_data_final_15m = id_data_final.resample("15min").ffill()

    # Use Queries to filter the data to the required year
    id_data_year_15m = id_data_final_15m.query(f"index < '{year+1}-01-01 00:00:00'")
    id_data_year_15m =id_data_year_15m.query(f"index >= '{year}-01-01 00:00:00'")
    id_price = id_data_year_15m["Price"]
    
    return id_price




if __name__ == '__main__':
    for i in range(2018, 2021):
        create_markets_info(i, save_csv=True)
    f = create_markets_info(year=2021, mean_da=75, mean_id=60, save_csv=True)
    print(f.head(10))
    create_markets_info(
        year=2030,
        mean_da=75,
        mean_id=60,
        fb=75,
        fp=80,
        save_csv=True)
