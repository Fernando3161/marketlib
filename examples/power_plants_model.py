'''
Created on 14.01.2022

@author: Fernando Penaherrera @UOL/OFFIS
@review: Steffen Wehkamp @OFFIS

This set of functions model the different power plants and their outputs
in the different markets

#1 Define Power Plants Scenarios
#2 Define Market Scenarios
#3 Build  Energy Systems which consider the different Scenarios
#4 Combine Scenarios and present results in a nice dataframe/csv
#5 Dump the scenarios as .oemof files
#6 Save results as graphics
'''

import sys
import os
path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path)


from src.electricity_markets.electricity_market_constraints import build_model_and_constraints
from src.electricity_markets.real_market_price_generator import create_markets_info
import seaborn as sns
import logging
from os.path import join
from examples.district_model_4_markets import get_district_dataframe,\
    solve_model, post_process_results
import matplotlib.pyplot as plt
from examples.common import EXAMPLES_PLOTS_DIR, EXAMPLES_RESULTS_DIR, check_and_create_all_folders
from src.electricity_markets.common import REAL_RAW_DATA_DIR
import pandas as pd
from oemof.solph.components import Sink, Source
from oemof.solph import (EnergySystem, Bus, Flow)
from enum import Enum

sns.set_style("darkgrid")
sns.set(font="arial")


class PowerPlants(Enum):
    '''
    Listing of the different power plants available
    '''
    COAL = 1
    GAS = 2
    BIOGAS = 3
    PV = 4
    WIND = 5
    BIOMASS = 6


def get_boundary_data(year=2020, days=366):
    """
    Constructs dataframes with the information for modeling.

    Args:
        year (int, optional): Year under consideration. Default is 2020.
        days (int, optional): Days to model. Default is 366 for a leap year.

    Returns:
        tuple: Two dataframes - district_df and market_data.
    """
    district_df = get_district_dataframe(year=year).head(24 * 4 * days)

    # Create Energy System with the dataframe time series
    market_data = create_markets_info(
        year=year, save_csv=False).head(
        days * 24 * 4)

    return district_df, market_data


def create_energy_system(scenario, district_df, market_data):
    """
    Creates an oemof energy system for the input scenario.

    Args:
        scenario (PowerPlants): One of the PowerPlants Scenario.
        district_df (pd.DataFrame): Dataframe with the district information.
        market_data (pd.DataFrame): Dataframe with market prices for each market.
    """

    meta_data = {}

    # Variable costs information, EUR/MWh
    PLANT_DATA_FILE = join(REAL_RAW_DATA_DIR, "Plant_Assumptions.xlsx")
    plant_price_data = pd.read_excel(PLANT_DATA_FILE, "Plant Assumptions",
                                     # index_col="Year",
                                     engine='openpyxl',
                                     skiprows=6, nrows=10)

    plant_price_data.drop([0], inplace=True)
    plant_price_data.rename(columns={"Unnamed: 2": "Technology"}, inplace=True)
    plant_price_data.set_index("Technology", inplace=True)
    meta_data["cv"] = {"coal": plant_price_data.at["Hard Coal", "Total c_V"],
                       "gas": plant_price_data.at["Natural Gas", "Total c_V"],
                       "biogas": plant_price_data.at["Biogas", "Total c_V"],
                       "pv": 0,
                       "wind": 0,
                       "biomass": plant_price_data.at["Biomass", "Total c_V"],
                       }

    # Max energy values for Renewables based on Installed capacity of 1MW and
    # real production as a fraction of 1MW
    meta_data["max_energy"] = {
        "coal": 1,  # MW
        "gas": 1,  # MW
        "biogas": 1,  # MW
        "wind": district_df["Wind_pu"].values,  # MW
        "pv": district_df["PV_pu"].values,  # MW
        "biomass": 1,  # MW
    }

    energy_system = EnergySystem(timeindex=district_df.index)

    label = scenario.name.lower()

    # create Bus
    b_el = Bus(label="b_el_out")

    # create Source
    source = Source(label="source", outputs={b_el: Flow(
        nominal_value=1,
        max=meta_data["max_energy"][label],
        variable_costs=meta_data["cv"][label])})

    # The markets each are modelled as a sink
    s_day_ahead = Sink(
        label="s_da",
        inputs={b_el: Flow(variable_costs=-market_data["day_ahead"].values)})

    s_intraday = Sink(
        label="s_id",
        inputs={b_el: Flow(variable_costs=-market_data["intra_day"].values)})

    s_future_base = Sink(
        label="s_fb",
        inputs={b_el: Flow(variable_costs=-market_data["future_base"].values)})

    s_future_peak = Sink(
        label="s_fp",
        inputs={b_el: Flow(variable_costs=-market_data["future_peak"].values
                           )})

    energy_system.add(
        b_el,
        source,
        s_day_ahead,
        s_future_base,
        s_future_peak,
        s_intraday)

    return energy_system


def calculate_kpis(results, market_data):
    """
    Calculate a set of KPIs and return them as a dataframe.

    Args:
        results (pd.DataFrame): Results dataframe.
        market_data (pd.DataFrame): Market dataframe.

    Returns:
        pd.Series: A series containing calculated KPIs.
    """

    total_energy = results.sum() / 4  # Since it it was in 15min intervals
    leng = min(len(results["b_el_out, s_da"]), len(market_data["day_ahead"]))
    income = {
        "income, da": results["b_el_out, s_da"][:leng].values *
        market_data["day_ahead"][:leng].values,
        "income, id": results["b_el_out, s_id"][:leng].values *
        market_data["intra_day"][:leng].values,
        "income, fb": results["b_el_out, s_fb"][:leng].values *
        market_data["future_base"][:leng].values,
        "income, fp": results["b_el_out, s_fp"][:leng].values *
        market_data["future_peak"][:leng].values}

    income["income, total"] = income["income, da"] + \
        income["income, id"] + income["income, fb"] + income["income, fp"]

    income_total = {k: round(v.sum() / 4, 1) for k, v in income.items()}
    income_total["average_price EUR/MWh"] = income_total["income, total"] / \
        total_energy["source, b_el_out"]
    # income_total = pd.Series(income_total)

    kpis = {}
    for idx, value in total_energy.items():
        kpis[idx] = value
    for idx, value in income_total.items():
        kpis[idx] = value

    kpis = pd.Series(kpis)
    return kpis


def model_power_plant_scenario(scenario, district_df, market_data, days=365):
    """
    Model a scenario and calculate KPIs based on the given boundary data.

    Args:
        scenario (PowerPlants): Scenario from PowerPlants.
        district_df (pd.DataFrame): Dataframe with information of the District.
        market_data (pd.DataFrame): Market Data with electricity price information.
        days (int, optional): Number of days to model, starting on 01/01. Default is 365.

    Returns:
        tuple: Two dataframes - results and kpis.
    """

    es = create_energy_system(scenario, district_df, market_data)
    model = build_model_and_constraints(es)
    solved_model = solve_model(model)
    results = post_process_results(solved_model)
    kpis = calculate_kpis(results, market_data)

    return results, kpis


def solve_and_write_data(year=2020, days=365):
    """
    Solve the different scenarios and write the data to an XLSX.

    Args:
        year (int, optional): Year of data. Default is 2020.
        days (int, optional): Number of days to model, starting on 01/01. Default is 365.

    Returns:
        dict: A dictionary containing results for different scenarios.
    """
    data_path = join(EXAMPLES_RESULTS_DIR, f'PowerPlantsModels_{year}.xlsx')
    writer = pd.ExcelWriter(data_path, engine='xlsxwriter')

    results_dict = {}

    # One cannot open and close the workbook w/o deleting previous books
    district_df, market_data = get_boundary_data(year=year, days=days)

    for scenario in PowerPlants:
        results, kpis = model_power_plant_scenario(
            scenario, district_df, market_data, days=days)

        results_dict[scenario] = results

        # Labels for spreadsheets
        ts_name = scenario.name + '-TimeSeries'
        kpi_name = scenario.name + '-KPIs'

        # Open Excel Writer
        workbook = writer.book
        worksheet = workbook.add_worksheet(ts_name)
        writer.sheets[ts_name] = worksheet

        # Save results and kpis to excel
        results.to_excel(writer, sheet_name=ts_name)
        worksheet = workbook.add_worksheet(kpi_name)
        writer.sheets[kpi_name] = worksheet
        kpis.to_excel(writer, sheet_name=kpi_name)

    writer.close()
    logging.info(f"Results and KPIs saved to {data_path}")
    return results_dict


def create_graphs(results_dict, year,days):
    """
    Create graphs and save them in the Results visualization directory.

    Args:
        results_dict (dict): Dictionary with the results from the different scenarios.
        year (int): Year for which the graphs are created.
    """
    for scenario in PowerPlants:
        results = results_dict[scenario]
        c = [c for c in results.columns if "b_el_out" in c.split(",")[0]]
        fig, ax = plt.subplots(figsize=(12, 4))
        results[c].plot(ax=ax)
        ax.set_title(str(scenario.name.capitalize()) + " Power Plant", fontweight="bold")
        fig.savefig(
            join(
                EXAMPLES_PLOTS_DIR,
                f"PowerPlant-{scenario.name.capitalize()}-{year}-{days}d.jpg"))
        logging.info(f"Plot saved for Scenario {scenario.name.capitalize()}")


def main(year=2020, days=365):
    """
    Chain functions to solve, write, and plot data from the scenario results.

    Args:
        year (int, optional): Year of data. Default is 2020.
        days (int, optional): Number of days to plot, starting on 01/01. Default is 365.
    """
    check_and_create_all_folders()
    results_dict = solve_and_write_data(year=year, days=days)
    create_graphs(results_dict, year,days=days)


if __name__ == '__main__':
    main(year=2020, days=7)
