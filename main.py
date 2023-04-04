import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from config import Config
from preparing_data import DataPreparation

class MainControl:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = Config("config.json")
        self.data_preparation = DataPreparation(args.data_file, self.config)

    def execute(self):
        self.data_preparation.read_data()
        self.data_preparation.data_formatting()
        self.data_preparation.print_correlation_matrix(self.data_preparation.df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aaa")
    parser.add_argument('-d', '--data_file', required=True)
    args = parser.parse_args()
    x = MainControl(args.data_file)
    x.execute()
