import pandas as pd
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import pandas as pd
import category_encoders as ce
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score


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

# fill in Nan -Values with mean
data = encoder.fit_transform(allData)
mean_value = data.mean()
data = data.fillna(mean_value)


# Compute the correlation matrix
correlation_matrix = data.corr()

# Get the most correlated features and throw away the ones that don't have enough correlation
power_cons_corr = (
    correlation_matrix["power_consumption"].abs().sort_values(ascending=False)
)
top_correlated_features = power_cons_corr[1:40].index.tolist()

# get the data of the features with the most correlation
x_filtered = data[top_correlated_features]

# split the set
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x_filtered, y, test_size=0.11, random_state=22
)


# K Nearest neighbors regression
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)


# print(neigh.score(X_train, y_train))
# print(neigh.score(X_test, y_test))


def leader_board_predict_fn(values):

    # encode the values
    values = encoder.fit_transform(values)

    # only get the most features that were mostly correlated to the model
    values_filtered = values[top_correlated_features]

    return neigh.predict(values_filtered)


def get_score():
    """
    Function to compute scores for train and test datasets (for the hidden training set).
    """

    # test score on train data
    try:
        y_predicted = leader_board_predict_fn(X_train)
        dataset_score = sklearn.metrics.mean_absolute_error(y_train, y_predicted)
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - dataset_score / dataset_mean) * 100
        print(dataset_accuracy)
        print("train score:" + str(explained_variance_score(y_train, y_predicted)))

    except Exception as e:
        print(e)
        dataset_score = float("nan")
    print(f"Train Dataset Score: {dataset_score}")

    # test score on test data
    try:
        y_predicted = leader_board_predict_fn(X_test)
        hiddendataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
        print("train test score:" + str(explained_variance_score(y_test, y_predicted)))
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - hiddendataset_score / dataset_mean) * 100
        print(dataset_accuracy)
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
        dataset_mean = y_test.mean().values[0]
        dataset_accuracy = (1 - hiddendataset_score / dataset_mean) * 100
        print(dataset_accuracy)
        print(
            "train score:"
            + str(sklearn.metrics.explained_variance_score(b, y_predicted))
        )

        print(f"Test Dataset Score: {hiddendataset_score}")
    except Exception as e:
        err = str(e)
        print(err)


get_score()
