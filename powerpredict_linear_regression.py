import pandas as pd
import os
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import os
import pwd
import time
import datetime
import pandas as pd
import category_encoders as ce


DATASET_PATH = "powerpredict_dataset.csv"

powerpredict = pd.read_csv(DATASET_PATH)
X = powerpredict.drop(columns=["power_consumption"])
y = powerpredict[["power_consumption"]]
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.33, random_state=42
)


def drop_object_columns(df):
    drop_cols = [
        c for t, c in zip([t != "object" for t in df.dtypes], df.columns) if not t
    ]
    return df.drop(columns=drop_cols)


DOC = drop_object_columns

allData = pd.concat([X_train, y_train], axis=1)

# with this we can encode the so called "string- features"
# we use this function instead of DOC, because we do not want to drop columns
encoder = ce.OrdinalEncoder()

# Encode the categorical variables
data = encoder.fit_transform(allData)
mean_value = data.mean()
data = data.fillna(mean_value)

# Compute the correlation matrix
correlation_matrix = data.corr()

power_cons_corr = (
    correlation_matrix["power_consumption"].abs().sort_values(ascending=False)
)
top_correlated_features = power_cons_corr[1:36].index.tolist()

x_filtered = data[top_correlated_features]


# linear regression with polynomial features
pol_reg_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
pol_reg_model.fit(x_filtered, y_train)
print(pol_reg_model.score(x_filtered, y_train))


def leader_board_predict_fn(values):

    # Encode the categorical variables
    values = encoder.fit_transform(values)

    values_filtered = values[top_correlated_features]

    return pol_reg_model.predict(values_filtered)


def get_values():
    """
    Function to compute scores for train and test datasets (for the real training set).
    """

    try:
        y_predicted = leader_board_predict_fn(x_filtered)
        dataset_score = sklearn.metrics.mean_absolute_error(y_train, y_predicted)
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - dataset_score / dataset_mean) * 100
        print(dataset_accuracy)
        print(
            "score normal train 1:"
            + str(sklearn.metrics.explained_variance_score(y_train, y_predicted))
        )
    except Exception as e:
        print(e)
        dataset_score = float("nan")
    print(f"Train Dataset Score: {dataset_score}")

    try:
        y_predicted = leader_board_predict_fn(X_test)
        hiddendataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - hiddendataset_score / dataset_mean) * 100
        print(dataset_accuracy)
        print(
            "score normal train 2:"
            + str(sklearn.metrics.explained_variance_score(y_test, y_predicted))
        )

        print(f"Test Dataset Score: {hiddendataset_score}")

    except Exception as e:
        err = str(e)
        print(err)


def get_score():
    """
    Function to compute scores for train and test datasets (for the hidden training set).
    """

    try:

        test_data = pd.read_csv(DATASET_PATH)
        X_test = test_data.drop(columns=["power_consumption"])
        y_test = test_data[["power_consumption"]]

        y_predicted = leader_board_predict_fn(X_test)
        dataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - dataset_score / dataset_mean) * 100
        print(dataset_accuracy)

        print(
            "score normal score 1:"
            + str(sklearn.metrics.explained_variance_score(y_test, y_predicted))
        )

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
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - hiddendataset_score / dataset_mean) * 100
        print(dataset_accuracy)

        print(
            "score normal score 2:"
            + str(sklearn.metrics.explained_variance_score(y_test, y_predicted))
        )

        print(f"Test Dataset Score: {hiddendataset_score}")
    except Exception as e:
        err = str(e)
        print(err)


get_values()
get_score()
