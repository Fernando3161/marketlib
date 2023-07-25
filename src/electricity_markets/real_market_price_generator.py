'''
Created on 20.01.2022

@author: Fernando Penaherrera @UOL/OFFIS
@author: Steffen Wehkamp OFFIS

Create market price time series from historical data
Using price pattern generated from historical data from 2015-2019

'''
import random
import sys
import os
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir,os.pardir))
sys.path.append(path)
print(path)
from src.electricity_markets.lehmer import LehmerRandom
import pandas as pd
from os.path import join
import logging
import os
from src.electricity_markets.common import PROC_DATA_DIR, RAW_DATA_DIR, REAL_RAW_DATA_DIR
from pandas.errors import SettingWithCopyWarning
import warnings
from scipy.stats import norm

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
    conversion= {0:2, 1:4,2:4,3:4,4:6,5:7,6:1}
    return conversion[x]

def create_price_pattern(year, market, mean_val=None, normalization = True):
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
        probabilities = [random_generator.random() for _ in range(4*len(market_df["Date"]))]
        market_df["Probability"] = probabilities[::4]


    if market == "id":
        market_df["Probability"] = [random_generator.random() for _ in range(len(market_df["Date"]))]
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
    price_data.drop(["Raw", "END"],axis=1, inplace=True)
    price_data=price_data.head(60)


    ## read volatility data
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
    volatility_data.drop(["END"],axis=1, inplace=True)
    volatility_data.dropna(axis=1, how="all", inplace=True)

    volatility_data=volatility_data.head(60)

    price_info ={
        "price": price_data,
        "volatility": volatility_data
    }

    for key, price_data in price_info.items():
        i = 0
        price = []
        while i < periods:
            month = market_df.at[i, "Month"]
            day_of_week = market_df.at[i, "EquivalentDayOfWeek"]  # Python puts monday at 0
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

    market_df["price_uc"]=norm.ppf(market_df["Probability"], loc=market_df["price"], scale=market_df["volatility"])


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
        market_df["market_price_year"] = market_df["market_price"] * mean_val / 100
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
        save_csv=True):
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
    return markets_data


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
