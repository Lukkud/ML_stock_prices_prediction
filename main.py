import argparse
import time
from config import Config
from preparing_data import DataPreparation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


class MainControl:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = Config("config.json")
        self.preparation = DataPreparation(args.data_file, self.config)

    def execute(self):
        # DataPreparation.print_correlation_matrix(self.preparation.df.drop(["Wolumen", "Zamkniecie", "Otwarcie", "Najwyzszy", "Najnizszy"], axis=1))
        DataPreparation.print_correlation_matrix(self.preparation.df)

        # self.model_estimation(RandomForestClassifier(), "Random Forest", self.config.parameter("rf_param_grid"), 5)
        # self.model_estimation(SVC(kernel='linear'), "SVC", self.config.parameter("svc_param_grid"), 5)
        # self.model_estimation(LogisticRegression(max_iter=1000), "Logistic regression", self.config.parameter("lr_param_grid"), 5)
        # self.model_estimation(BaggingClassifier(), "Bagging", cv=10)
        self.model_estimation(AdaBoostClassifier(), "AdaBoost", self.config.parameter("adaboost_grid"), cv=5)

    def model_estimation(self, model, model_name, param_grid={}, cv=None):
        model.fit(self.preparation.X_train, self.preparation.y_train)

        # Use grid search to find the best hyperparameters
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv)

        # Fit the random search object to the data
        grid_search.fit(self.preparation.X_train, self.preparation.y_train)
        best_model = grid_search.best_estimator_

        # Print the best hyperparameters
        print(f'Best hyperparameters - {model_name}: ', grid_search.best_params_)

        # Print confucion matrix
        print(f'Confucion matrix (train) - {model_name}: \n',
              confusion_matrix(self.preparation.y_train, best_model.predict(self.preparation.X_train)))
        print(f'Confucion matrix (test) - {model_name}: \n',
              confusion_matrix(self.preparation.y_test, best_model.predict(self.preparation.X_test)))

        #Cross validation
        cv_score_train = cross_val_score(best_model, self.preparation.X_train, self.preparation.y_train, cv=cv)
        print(f'Cross validation (train) - {model_name}: ', np.mean(cv_score_train))
        cv_score_test = cross_val_score(best_model, self.preparation.X_test, self.preparation.y_test, cv=cv)
        print(f'Cross validation (test) - {model_name}: ', np.mean(cv_score_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aaa")
    parser.add_argument('-d', '--data_file', required=True)
    args = parser.parse_args()
    tic = time.time()
    x = MainControl(args.data_file)
    x.execute()
    toc = time.time()
    print("\nExecution time: {}". format(toc - tic))
