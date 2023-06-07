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
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import dummy

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


def predict_show_metrics(name, reg, metric):
    print(f"{name}", metric(y, reg.predict(DOC(X))))


reg = sklearn.dummy.DummyRegressor()
reg.fit(DOC(X), y)
metric = sklearn.metrics.mean_absolute_error

predict_show_metrics("Dummy", reg, metric)


# This can be changed
# Polynomial regression model


# get features with the highest correlation
power_cons_corr = DOC(powerpredict).corr()["power_consumption"]
power_cons_corr_abs = abs(power_cons_corr).sort_values(ascending=False)[1:]
highest_corr = power_cons_corr_abs[0:20].keys()
x_filtered = powerpredict[highest_corr]
print(x_filtered.shape)


# linear regression with polynomial
pol_reg_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
pol_reg_model.fit(x_filtered, y)

# tree regression
model = DecisionTreeRegressor()
model.fit(x_filtered, y)
print("model fitted")

# Train Dataset Score: 3061.6933126681356
# Test Dataset Score: 3181.6676929868504
# Train Dataset Score: 17.286747383246727
# Test Dataset Score: 2987.4145525455483

# random=44,
# Train Dataset Score: 826.6064280458045
# Test Dataset Score: 3034.3093991967244


def leader_board_predict_fn(values):
    print(values.shape)

    values = DOC(values)
    # YOUR CODE HERE (please remove 'raise NotImplementedError()')
    x_filtered = values[highest_corr]

    return pol_reg_model.predict(x_filtered)  # replace this with your implementation


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
    except Exception:
        dataset_score = float("nan")
    print(f"Train Dataset Score: {dataset_score}")

    import os
    import pwd
    import time
    import datetime
    import pandas as pd

    user_id = pwd.getpwuid(os.getuid()).pw_name
    curtime = time.time()
    dt_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    try:
        HIDDEN_DATASET_PATH = os.path.expanduser("/data/mlproject22-test-data/")
        test_data = pd.read_csv(
            os.path.join(HIDDEN_DATASET_PATH, "hidden_powerpredict.csv.zip")
        )
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
