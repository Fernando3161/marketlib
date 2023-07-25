'''
Created on 20.05.2021

@author: Fernando Penaherrera @UOL/OFFIS
'''
import os
from os.path import join


def get_project_root():
    """
    Return the path to the project root directory.

    Returns:
        str: A directory path.
    """
    return os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        os.pardir,
    ))


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


BASE_DIR = get_project_root()
SOURCE_DIR = join(BASE_DIR, "src")
ENERGY_MARKETS_DIR = join(SOURCE_DIR, "energy_markets")
EXAMPLES_DIR = join(BASE_DIR, "examples")
EXAMPLES_DATA_DIR = join(EXAMPLES_DIR, "data")
EXAMPLES_RESULTS_DIR = join(EXAMPLES_DIR, "results")
EXAMPLES_PLOTS_DIR = join(EXAMPLES_DIR, "plots")


def check_and_create_all_folders():
    """
    Check and create all the necessary project folders if missing.
    """
    folders_to_check = [
        BASE_DIR,
        SOURCE_DIR,
        ENERGY_MARKETS_DIR,
        EXAMPLES_DIR,
        EXAMPLES_DATA_DIR,
        EXAMPLES_RESULTS_DIR,
        EXAMPLES_PLOTS_DIR,
    ]
    check_and_create_folders(folders_to_check)


if __name__ == '__main__':
    a = 1

    print(EXAMPLES_PLOTS_DIR)
    check_and_create_all_folders()
