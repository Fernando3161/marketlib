'''
Created on 20.05.2021

@author: Fernando Penaherrera @UOL/OFFIS
'''
import os
from os.path import join


def get_project_root():
    """Return the path to the project root directory.

    :return: A directory path.
    :rtype: str
    """
    return os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    ))


BASE_DIR = get_project_root()
SOURCE_DIR = join(BASE_DIR, "src")
ELECTRICITY_MARKETS_DIR = join(BASE_DIR, "electricity_markets")
RAW_DATA_DIR = join(ELECTRICITY_MARKETS_DIR, "raw")
REAL_RAW_DATA_DIR = join(RAW_DATA_DIR, "real_data")
DATA_DIR = join(BASE_DIR, "data")
PROC_DATA_DIR = join(DATA_DIR, "processed")
MODEL_DATA_DIR = join(DATA_DIR, "dumped_models")
RESULTS_DIR = join(BASE_DIR, "results")
RESULTS_DATA_DIR = join(RESULTS_DIR, "data")
RESULTS_VIS_DIR = join(RESULTS_DIR, "visualizations")


def check_and_create_folders(folders):
    """
    Check if the specified folders exist; if not, create them.

    Args:
        folders (list[str]): A list of folder paths to check and create if missing.
    """
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder '{folder}' created.")
        else:
            pass
            # print(f"Folder '{folder}' already exists.")


def check_and_create_all_folders():
    """
    Check and create all the necessary project folders if missing.
    """
    folders_to_check = [
        BASE_DIR,
        SOURCE_DIR,
        ELECTRICITY_MARKETS_DIR,
        RAW_DATA_DIR,
        REAL_RAW_DATA_DIR,
        DATA_DIR,
        PROC_DATA_DIR,
        MODEL_DATA_DIR,
        RESULTS_DIR,
        RESULTS_DATA_DIR,
        RESULTS_VIS_DIR,
    ]
    check_and_create_folders(folders_to_check)


if __name__ == '__main__':
    pass
