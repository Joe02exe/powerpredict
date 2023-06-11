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
import os
import pwd
import time
import datetime
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce


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
data = encoder.fit_transform(allData)

# Compute the correlation matrix
correlation_matrix = data.corr()

power_cons_corr = (
    correlation_matrix["power_consumption"].abs().sort_values(ascending=False)
)
top_correlated_features = power_cons_corr[1:50].index.tolist()

x_filtered = data[top_correlated_features]
x_filtered = x_filtered.dropna()


# linear regression with polynomial
# pol_reg_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
# pol_reg_model.fit(x_filtered, y)
polynomial_features = PolynomialFeatures(degree=2)
x_polynomial = polynomial_features.fit_transform(x_filtered)
# print("Shape of x_filtered:", x_filtered.shape)
# print("Shape of x_polynomial:", x_polynomial.shape)
# print(y.shape)
ridgeRegression = Ridge()
ridgeRegression.fit(x_filtered, y)


# model = DecisionTreeRegressor(min_samples_split=2)
# model.fit(x_filtered, y)


def leader_board_predict_fn(values):

    # Encode the categorical variables
    values = encoder.fit_transform(values)

    values_filtered = values[top_correlated_features]

    return ridgeRegression.predict(
        values_filtered
    )  # replace this with your implementation


def get_score():
    """
    Function to compute scores for train and test datasets.
    """

    try:

        test_data = pd.read_csv(DATASET_PATH)
        X_test = test_data.drop(columns=["power_consumption"])
        y_test = test_data[["power_consumption"]]

        y_predicted = leader_board_predict_fn(X_test)
        dataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
    except Exception as e:
        print(e)
        dataset_score = float("nan")
    print(f"Train Dataset Score: {dataset_score}")

    user_id = pwd.getpwuid(os.getuid()).pw_name
    curtime = time.time()
    dt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        test_data = pd.read_csv("hidden_powerpredict.csv")
        X_test = test_data.drop(columns=["power_consumption"])
        y_test = test_data[["power_consumption"]]
        y_predicted = leader_board_predict_fn(X_test)
        hiddendataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
        print(f"Test Dataset Score: {hiddendataset_score}")
        score_dict = dict(
            score_hidden=hiddendataset_score,
            score_train=dataset_score,
            unixtime=curtime,
            user=user_id,
            dt=dt_now,
            comment="",
        )
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

    # if list(pathlib.Path(os.getcwd()).parents)[0].name == 'source':
    #    print("we are in the source directory... replacing values.")
    #    print(pd.DataFrame([score_dict]))
    #    score_dict["score_hidden"] = -1
    #    score_dict["score_train"] = -1
    #    print("new values:")
    #    print(pd.DataFrame([score_dict]))

    pd.DataFrame([score_dict]).to_csv("powerpredict.csv", index=False)


get_score()
