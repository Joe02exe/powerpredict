import pandas as pd
import os
import sklearn
import sklearn.metrics
import pathlib
import sklearn.model_selection
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import dummy
import warnings
import os
import pwd
import time
import datetime
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, explained_variance_score


DATASET_PATH = "powerpredict_dataset.csv"

powerpredict = pd.read_csv(DATASET_PATH)
X = powerpredict.drop(columns=["power_consumption"])
y = powerpredict[["power_consumption"]]


def drop_object_columns(df):
    drop_cols = [
        c for t, c in zip([t != "object" for t in df.dtypes], df.columns) if not t
    ]
    return df.drop(columns=drop_cols)


DOC = drop_object_columns

allData = pd.concat([X, y], axis=1)

# with this we can encode the so called "string- features"
# we use this function instead of DOC, because we do not want to drop columns
encoder = ce.OrdinalEncoder()

# Encode the categorical variables
warnings.filterwarnings(
    "ignore",
    message="No categorical columns found. Calling 'transform' will only return input data.",
    category=UserWarning,
)
data = encoder.fit_transform(allData)
warnings.resetwarnings()


# Compute the correlation matrix
correlation_matrix = data.corr()

power_cons_corr = (
    correlation_matrix["power_consumption"].abs().sort_values(ascending=False)
)
top_correlated_features = power_cons_corr[1:40].index.tolist()

x_filtered = data[top_correlated_features]
x_filtered = x_filtered.dropna()

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x_filtered, y, test_size=0.11, random_state=22
)


# linear regression with polynomial

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)
print(neigh.score(X_train, y_train))
print(neigh.score(X_test, y_test))


# polynomial_features = PolynomialFeatures(degree=2)
# x_polynomial = polynomial_features.fit_transform(x_filtered)
# print("Shape of x_filtered:", x_filtered.shape)
# print("Shape of x_polynomial:", x_polynomial.shape)
# print(y.shape)
# ridgeRegression = Ridge()
# ridgeRegression.fit(x_filtered, y)


# polynomial 3:
# Train Dataset Score: 2063.555664003627
# Test Dataset Score: 3257.2906929313135

# model = DecisionTreeRegressor(min_samples_split=2)
# model.fit(x_filtered, y)


def leader_board_predict_fn(values):

    # Encode the categorical variables
    warnings.filterwarnings(
        "ignore",
        message="No categorical columns found. Calling 'transform' will only return input data.",
        category=UserWarning,
    )

    values = encoder.fit_transform(values)
    warnings.resetwarnings()

    values_filtered = values[top_correlated_features]

    return neigh.predict(values_filtered)


def get_score():
    """
    Function to compute scores for train and test datasets (for the hidden training set).
    """

    try:
        y_predicted = leader_board_predict_fn(X_train)
        dataset_score = sklearn.metrics.mean_absolute_error(y_train, y_predicted)
        print("train score:" + str(explained_variance_score(y_train, y_predicted)))

    except Exception as e:
        print(e)
        dataset_score = float("nan")
    print(f"Train Dataset Score: {dataset_score}")

    user_id = pwd.getpwuid(os.getuid()).pw_name
    curtime = time.time()
    dt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # test score on train data
    try:
        y_predicted = leader_board_predict_fn(X_test)
        hiddendataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
        print("train test score:" + str(explained_variance_score(y_test, y_predicted)))

        print(f"Test Dataset Score: {hiddendataset_score}")
    except Exception as e:
        err = str(e)
        print(err)

    # test on test csv
    try:
        test_data = pd.read_csv("hidden_powerpredict.csv")
        a = test_data.drop(columns=["power_consumption"])
        b = test_data[["power_consumption"]]
        y_predicted = leader_board_predict_fn(a)
        hiddendataset_score = sklearn.metrics.mean_absolute_error(b, y_predicted)
        print(
            "train score:"
            + str(sklearn.metrics.explained_variance_score(b, y_predicted))
        )

        print(f"Test Dataset Score: {hiddendataset_score}")
    except Exception as e:
        err = str(e)
        print(err)
        score_dict = dict(
            score_hidden=float("nan"),
            score_train=dataset_score,
            unixtime=curtime,
            user=user_id,
            dt=dt_now,
            comment=err,
        )


get_score()
