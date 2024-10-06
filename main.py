import argparse
import time
from pathlib import Path
import os
from config import Config
from preparing_data import DataPreparation

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import RocCurveDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")


DIR_PATH = Path(os.path.abspath(__file__)).parents[0]
PLOT_PATH = os.path.join(DIR_PATH, 'plot_files')
DATA_PATH = os.path.join(DIR_PATH, 'data_files')
Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

class MainControl:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = Config("config.json")
        self.preparation = DataPreparation(args.data_file, self.config)

    def oob_plot(self, min_estimators, max_estimators, X, y):
        # model = BaggingClassifier(DecisionTreeClassifier(criterion="entropy", min_samples_split=2, min_samples_leaf=21, max_depth=8), oob_score=True)
        model = RandomForestClassifier(max_features="log2", n_estimators=220, n_jobs=-1, criterion="entropy",
                                       min_samples_split=2, min_samples_leaf=21, max_depth=8, ccp_alpha=0, oob_score=True)
        oob_score_list = []
        for i in range(min_estimators, max_estimators + 1, 5):
            params_dict = {"n_estimators": i}
            model.set_params(**params_dict)
            model.fit(X, y)
            oob_score_list.append(1 - model.oob_score_)

        plt.plot(range(min_estimators, max_estimators + 1, 5), oob_score_list)
        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("Błąd OOB")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, f'oob_rf.png'), format='png')
        plt.show()

    def model_estimation(self, model, model_name, param_grid={}, cv=None):
        model.fit(self.preparation.X_train, self.preparation.y_train)

        # Use grid search to find the best hyperparameters
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv)

        # Fit the random search object to the data
        grid_search.fit(self.preparation.X_train, self.preparation.y_train)
        best_model = grid_search.best_estimator_

        # Print the best hyperparameters
        print(f'Best hyperparameters - {model_name}: ', grid_search.best_params_)

        # Print confusion matrix
        print(f'Confusion matrix (train) - {model_name}: \n',
              confusion_matrix(self.preparation.y_train, best_model.predict(self.preparation.X_train)))
        print(f'Confusion matrix (test) - {model_name}: \n',
              confusion_matrix(self.preparation.y_test, best_model.predict(self.preparation.X_test)))

        #Cross validation
        cv_score_train_acc = cross_val_score(best_model, self.preparation.X_train, self.preparation.y_train, cv=cv)
        print(f'Cross validation Accuracy (train) - {model_name}: ', np.mean(cv_score_train_acc))
        cv_score_train_f1 = cross_val_score(best_model, self.preparation.X_train, self.preparation.y_train, cv=cv, scoring='f1')
        print(f'Cross validation F1 (train) - {model_name}: ', np.mean(cv_score_train_f1))
        cv_score_train_auc = cross_val_score(best_model, self.preparation.X_train, self.preparation.y_train, cv=cv, scoring='roc_auc')
        print(f'Cross validation AUC (train) - {model_name}: ', np.mean(cv_score_train_auc))
        cv_score_test_acc = cross_val_score(best_model, self.preparation.X_test, self.preparation.y_test, cv=cv)
        print(f'Cross validation Accuracy (test) - {model_name}: ', np.mean(cv_score_test_acc))
        cv_score_test_f1 = cross_val_score(best_model, self.preparation.X_test, self.preparation.y_test, cv=cv, scoring='f1')
        print(f'Cross validation F1 (test) - {model_name}: ', np.mean(cv_score_test_f1))
        cv_score_test_auc = cross_val_score(best_model, self.preparation.X_test, self.preparation.y_test, cv=cv, scoring='roc_auc')
        print(f'Cross validation AUC (test) - {model_name}: ', np.mean(cv_score_test_auc))

        result = permutation_importance(best_model, self.preparation.X_test, self.preparation.y_test, n_repeats=100)
        sorted_importances_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=self.preparation.X_test.columns[sorted_importances_idx],
        )

        plt.rcParams.update({'font.size': 10})
        ax = importances.plot.box(vert=False, whis=15)
        ax.set_title("Permutacyjna istotność parametrów (na zbiorze testowym)")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.set_xlabel("Spadek trafności po eliminacji zmiennej")
        plt.xlim(-0.05, 0.3)
        ax.figure.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, f'{model_name}_permutation_importance.png'), format='png')
        plt.show()

        RocCurveDisplay.from_estimator(best_model, self.preparation.X_test, self.preparation.y_test)
        plt.plot([0, 1], [0, 1], color = 'red', linewidth=1)
        plt.xlabel(r"1 - specyficzność")
        plt.ylabel(r"czułość")
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_PATH, f'{model_name}_roc_curve.png'), format='png')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aaa")
    parser.add_argument('-d', '--data_file', required=True)
    args = parser.parse_args()
    tic = time.time()
    x = MainControl(args.data_file)

    # DataPreparation.print_correlation_matrix(x.preparation.df)

    # x.model_estimation(DecisionTreeClassifier(), "Decision_Tree", x.config.parameter("dt_param_grid"), 5)
    # x.model_estimation(BaggingClassifier(estimator=DecisionTreeClassifier(criterion="entropy", min_samples_split=2, min_samples_leaf=21, max_depth=8)), "Bagging", x.config.parameter("bagging_grid"), cv=5)
    # x.model_estimation(RandomForestClassifier(), "Random_Forest", x.config.parameter("rf_param_grid"), 5)
    # x.model_estimation(AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion="entropy", min_samples_split=2, min_samples_leaf=21, max_depth=8)), "AdaBoost", x.config.parameter("adaboost_grid"), cv=5)
    # x.model_estimation(LogisticRegression(), "Logistic_regression", x.config.parameter("lr_param_grid"), 5)
    # x.model_estimation(SVC(), "SVC", x.config.parameter("svc_param_grid"), 5)

    # x.oob_plot(1, 500, x.preparation.X_test, x.preparation.y_test)

    toc = time.time()
    print("\nExecution time: {}". format(toc - tic))
