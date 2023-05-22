import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import ta
from ta import volume as ta_volume
from ta import trend as ta_trend
from ta import momentum as ta_momentum
from ta import volatility as ta_volatility


class DataPreparation:
    def __init__(self, file_name, config_file):
        self.file_name = file_name
        self.config = config_file
        self.resources_path = self.config.parameter("resources_path")
        self.test_size = self.config.parameter("test_size")
        self.shift = self.config.parameter("shift")
        self.df = pd.DataFrame({})

        self.read_data()
        self.data_formatting(self.config.parameter("filling_gaps"))
        self.add_ta_indexes()

        self.X = self.df.drop('y', axis=1)
        scaler = preprocessing.StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X))
        self.y = self.df['y']
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
        # corr_matrix.to_excel('D:\Pliki_latex\Praca_magisterska\corr_matrix.xlsx')
        plt.figure(figsize=(30, 20))
        sn.heatmap(corr_matrix, annot=True)
        plt.show()

    def add_ta_indexes(self):
        self.df["ACC_dist_line"] = ta_volume.acc_dist_index(close=self.df["Zamkniecie"],
                                                            volume=self.df["Wolumen"],
                                                            high=self.df["Najwyzszy"],
                                                            low=self.df["Najnizszy"])

        self.df["ADX"] = ta_trend.adx(close=self.df["Zamkniecie"],
                                      high=self.df["Najwyzszy"],
                                      low=self.df["Najnizszy"],
                                      window=20)

        self.df["RSI"] = ta_momentum.rsi(close=self.df["Zamkniecie"], window=20)
        self.df["ROC"] = ta_momentum.roc(close=self.df["Zamkniecie"], window=20)
        self.df["SMA"] = ta_trend.sma_indicator(close=self.df["Zamkniecie"],
                                                window=10)
        self.df["CCI"] = ta_trend.cci(close=self.df["Zamkniecie"],
                                      high=self.df["Najwyzszy"],
                                      low=self.df["Najnizszy"],
                                      window=10)
        self.df["Stoch"] = ta_momentum.stoch(high=self.df["Najwyzszy"],
                                             low=self.df["Najnizszy"],
                                             close=self.df["Zamkniecie"])
        self.df["MACD"] = ta_trend.macd(close=self.df["Zamkniecie"])
        self.df["Bollinger_hi"] = ta_volatility.bollinger_hband_indicator(close=self.df["Zamkniecie"])
        self.df["Bollinger_li"] = ta_volatility.bollinger_lband_indicator(close=self.df["Zamkniecie"])
        self.df["Keltner_hi"] = ta_volatility.keltner_channel_hband_indicator(close=self.df["Zamkniecie"],
                                                                              high=self.df["Najwyzszy"],
                                                                              low=self.df["Najnizszy"])
        self.df["Keltner_li"] = ta_volatility.keltner_channel_lband_indicator(close=self.df["Zamkniecie"],
                                                                              high=self.df["Najwyzszy"],
                                                                              low=self.df["Najnizszy"])

        self.df = self.df.drop(["Wolumen", "Zamkniecie", "Otwarcie", "Najwyzszy", "Najnizszy"], axis=1)
        self.df = self.df.dropna()
