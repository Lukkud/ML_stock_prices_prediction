import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


class DataPreparation:
    def __init__(self, file_name):
        self.file_name = file_name
        self.resources_path = r"D:\Studia II stopien\IV semestr\Praca\Dane"
        self.df = pd.DataFrame({})

    def read_data(self):
        """
        Reading data from xlsx or csv file
        :return: None as method only changes existing dataframe df
        """
        file_path = os.path.join(self.resources_path, self.file_name)
        if ".csv" in file_path:
            self.df = pd.read_csv(file_path)
        elif ".xlsx" in file_path:
            self.df = pd.read_excel(io=file_path, sheet_name="Sheet")
        else:
            print("Unsupported file extension")

    def data_preparation(self):
        """
        Filling gaps in df dataframe and formatting dates in column "Data"
        :return: None as method only changes existing dataframe df
        """
        self.df["Data"] = pd.to_datetime(self.df["Data"], format='%Y-%m-%d')
        date_index = pd.date_range(start=self.df["Data"].min(), end=self.df["Data"].max(), freq='D')
        self.df = self.df.set_index('Data')
        self.df = self.df.reindex(index=date_index, method="bfill")

    @staticmethod
    def print_correlation_matrix(df):
        """
        Creates correlation matrix for a given dataframe and plots it using matplotlib and seaborn
        :param df: Dataframe (each column must be numeric type!)
        :return: plot of correlation matrix
        """
        corrMatrix = df.corr()
        plt.figure(figsize=(15, 10))
        sn.heatmap(corrMatrix, annot=True)
        plt.show()

if __name__ == "__main__":
    x = DataPreparation("wig20_d.csv")
    x.read_data()
    x.data_preparation()
    x.print_correlation_matrix(x.df.drop(columns=["Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie", "Wolumen"]))
