import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.momentum import ROCIndicator


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

    def data_preparation_filling_gaps(self):
        """
        Filling gaps in df dataframe and formatting dates in column "Data"
        :return: None as method only changes existing dataframe df
        """
        self.df["Data"] = pd.to_datetime(self.df["Data"], format='%Y-%m-%d')
        date_index = pd.date_range(start=self.df["Data"].min(), end=self.df["Data"].max(), freq='D')
        self.df = self.df.set_index('Data')
        self.df = self.df.reindex(index=date_index, method="bfill")

    def data_preparation(self):
        """
        Filling gaps in df dataframe and formatting dates in column "Data"
        :return: None as method only changes existing dataframe df
        """
        self.df["Data"] = pd.to_datetime(self.df["Data"], format='%Y-%m-%d')
        self.df = self.df.set_index('Data')

    @staticmethod
    def print_correlation_matrix(df):
        """
        Creates correlation matrix for a given dataframe and plots it using matplotlib and seaborn
        :param df: Dataframe (each column must be numeric type!)
        :return: correlation matrix plot
        """
        corrMatrix = df.corr()
        plt.figure(figsize=(15, 10))
        sn.heatmap(corrMatrix, annot=True)
        plt.show()

    def execute(self):
        self.read_data()
        # self.data_preparation()
        self.data_preparation_filling_gaps()

        self.df["wolumen_diff"] = self.df["Wolumen"] - self.df["Wolumen"].shift(1)
        self.df["zamkniecie_df1"] = self.df["Zamkniecie"].shift(1)
        self.df["Diff"] = self.df["Zamkniecie"] - self.df["zamkniecie_df1"]
        self.df["Log_stopy"] = np.log(self.df["Zamkniecie"] / self.df["zamkniecie_df1"])
        self.df["Znak"] = self.df.apply(lambda row: 1 if row["Diff"] >= 0 else 0, axis=1)

        # Example of tachnical anaylsis indicators
        indicator_rsi = RSIIndicator(close=self.df["Zamkniecie"], window=20)
        self.df["RSI"] = indicator_rsi.rsi()
        indicator_roc = ROCIndicator(close=self.df["Zamkniecie"], window=20)
        self.df["ROC"] = indicator_roc.roc()
        # Removing empty rows
        self.df = self.df.dropna()
        # Print correlation matrix
        self.print_correlation_matrix(self.df.drop(columns=["Otwarcie", "Najwyzszy", "Najnizszy", "Zamkniecie", "Wolumen"]))


        X = self.df[["RSI", "ROC"]]
        y = self.df["Znak"]
        X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=0.2, random_state=290392)
        model = RandomForestClassifier(random_state=290392)
        model.fit(X_train, y_train)
        y_pred = model.predict(X)

        print("Accuracy dla treningowych: ", model.score(X_train, y_train))
        print("Accuracy dla testowych: ", model.score(X_test, y_test))

        #Cross validation
        from sklearn.model_selection import cross_val_score
        cv_score = cross_val_score(model, X_train, y_train, cv=10)
        print(np.mean(cv_score))

        #Print confusion matrix
        from sklearn import metrics
        cnf_matrix = metrics.confusion_matrix(y.values.ravel(), y_pred)
        print(cnf_matrix)

        # Print ROC and AUC metrics
        fpr, tpr, _ = metrics.roc_curve(y.values.ravel(), y_pred)
        auc = metrics.roc_auc_score(y.values.ravel(), y_pred)
        plt.plot(fpr, tpr)
        plt.title("Krzywa ROC")
        plt.show()

        print("AUC: ", auc)


if __name__ == "__main__":
    x = DataPreparation("cdr_d.csv")
    x.execute()
