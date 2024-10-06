import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ta import trend as ta_trend
from ta import momentum as ta_momentum
from ta import volatility as ta_volatility


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

class DataPreparation:
    def __init__(self, file_name, config_file):
        self.file_name = file_name
        self.config = config_file
        self.resources_path = DATA_PATH
        self.test_size = self.config.parameter("test_size")
        self.shift = self.config.parameter("shift")
        self.df = pd.DataFrame({})
        self.read_data()
        self.data_formatting(self.config.parameter("filling_gaps"))
        self.add_ta_indexes()
        self.X = self.df.drop('y', axis=1)
        self.y = self.df['y']

        scaler = preprocessing.StandardScaler()
        col_scaler_list = ["ADX", "RSI", "ROC", "EMA", "CCI", "Oscylator stoch.", "MACD"]
        col_scaler_dict = self.X[col_scaler_list].columns
        self.X[["ADX", "RSI", "ROC", "EMA", "CCI", "Oscylator stoch.", "MACD"]] = \
            pd.DataFrame(scaler.fit_transform(self.X[["ADX", "RSI", "ROC", "EMA", "CCI", "Oscylator stoch.", "MACD"]]),
                         index=self.X.index,
                         columns=col_scaler_dict)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=290392)

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

    def data_formatting(self, filling_gaps=False):
        """
        Filling gaps in df dataframe and formatting dates in column "Data"
        :param filling_gaps: (bool) If true missing data will be filled with next valid observation
        :return: None as method only changes existing dataframe df
        """
        self.df["Data"] = pd.to_datetime(self.df["Data"], format='%Y-%m-%d')
        if filling_gaps:
            date_index = pd.date_range(start=self.df["Data"].min(), end=self.df["Data"].max(), freq='D')
            self.df = self.df.set_index('Data')
            self.df = self.df.reindex(index=date_index, method="bfill")
        else:
            self.df = self.df.set_index('Data')
        self.df["y"] = self.df["Zamkniecie"] - self.df["Zamkniecie"].shift(1)
        self.df["y"] = self.df.apply(lambda row: 1 if row["y"] >= 0 else 0, axis=1)
        self.df["y"] = self.df['y'].shift(self.shift)

    @staticmethod
    def print_correlation_matrix(df):
        """
        Creates correlation matrix for a given dataframe and plots it using matplotlib and seaborn
        :param df: Dataframe (each column must be numeric type!)
        :return: plot of correlation matrix
        """
        corr_matrix = df.corr()
        plt.rcParams['font.size'] = 24
        corr_matrix.to_excel(os.path.join(DATA_PATH, f'corr_matrix.xlsx'))
        plt.figure(figsize=(30, 20))
        sn.heatmap(corr_matrix, annot=True)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(PLOT_PATH, f'corr_matrix.png'), format='png')
        plt.show()

    def add_ta_indexes(self):
        self.df["ADX"] = ta_trend.adx(close=self.df["Zamkniecie"],
                                      high=self.df["Najwyzszy"],
                                      low=self.df["Najnizszy"])
        self.df["RSI"] = ta_momentum.rsi(close=self.df["Zamkniecie"])
        self.df["ROC"] = ta_momentum.roc(close=self.df["Zamkniecie"])
        self.df["EMA"] = ta_trend.ema_indicator(close=self.df["Zamkniecie"])
        self.df["CCI"] = ta_trend.cci(close=self.df["Zamkniecie"],
                                      high=self.df["Najwyzszy"],
                                      low=self.df["Najnizszy"])
        self.df["Oscylator stoch."] = ta_momentum.stoch(high=self.df["Najwyzszy"],
                                                        low=self.df["Najnizszy"],
                                                        close=self.df["Zamkniecie"])
        self.df["MACD"] = ta_trend.macd(close=self.df["Zamkniecie"])
        self.df["Bollinger - gg"] = ta_volatility.bollinger_hband_indicator(close=self.df["Zamkniecie"])
        self.df["Bollinger - dg"] = ta_volatility.bollinger_lband_indicator(close=self.df["Zamkniecie"])

        self.df = self.df.drop(["Wolumen", "Zamkniecie", "Otwarcie", "Najwyzszy", "Najnizszy"], axis=1)
        self.df = self.df.dropna()
