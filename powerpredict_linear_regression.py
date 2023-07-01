import pandas as pd
import os
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


import os
import pandas as pd

N = 21

le = preprocessing.OrdinalEncoder()

DATASET_PATH = "powerpredict_dataset.csv"

powerpredict = pd.read_csv(DATASET_PATH)
X = powerpredict.drop(columns=["power_consumption"])
y = powerpredict[["power_consumption"]]


encoder = preprocessing.OrdinalEncoder()

# Encode the data without the first row
encoded_data = encoder.fit_transform(X)

# Create a DataFrame with the encoded data
data = pd.DataFrame(encoded_data)

# Fill NaN values with mean
mean_value = data.mean()
data = data.fillna(mean_value)

allData = pd.concat([data, y], axis=1)

# Compute the correlation matrix
correlation_matrix = allData.corr()

# Get the most correlated features and throw away the ones that don't have enough correlation
power_cons_corr = (
    correlation_matrix["power_consumption"].abs().sort_values(ascending=False)
)
selected_columns = power_cons_corr[1:N].index.tolist()

x_filtered = allData[selected_columns]

# split the set
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x_filtered, y, test_size=0.20, random_state=22
)
# linear regression
neigh = make_pipeline(PolynomialFeatures(2), LinearRegression())
neigh.fit(X_train, y_train)
y_predicted = neigh.predict(X_test)
y_predicted_train = neigh.predict(X_train)
from joblib import dump, load

dump(neigh, "linear_regression.joblib")

dataset_score = sklearn.metrics.mean_absolute_error(y_train, y_predicted_train)
print("Test score:" + str(dataset_score))
dataset_score = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
print("Train score:" + str(dataset_score))

cv_scores = cross_val_score(
    neigh, X_train, y_train, cv=10
)  # Adjust the number of folds as needed

# Print the cross-validation scores


from joblib import dump, load

dump(neigh, "linear_regression.joblib")


def leader_board_predict_fn(values):
    values = le.fit_transform(values)
    # only get the most features that were mostly correlated to the model
    values_filtered = values[:, selected_columns]
    from joblib import dump, load

    loaded_rfr = load(
        "linear_regression.joblib"
    )  # Provide the file path to the saved model
    return loaded_rfr.predict(values_filtered)


try:
    test_data = pd.read_csv("hidden_powerpredict.csv")
    a = test_data.drop(columns=["power_consumption"])
    b = test_data[["power_consumption"]]
    y_predicted = leader_board_predict_fn(a)
    hiddendataset_score = sklearn.metrics.mean_absolute_error(b, y_predicted)
    dataset_mean = y_test.mean().values[0]
    dataset_accuracy = (1 - hiddendataset_score / dataset_mean) * 100
    print("Accuracy:" + str(dataset_accuracy))
    print(
        "train score:" + str(sklearn.metrics.explained_variance_score(b, y_predicted))
    )

    print(f"Test Dataset Score: {hiddendataset_score}")
except Exception as e:
    err = str(e)
    print(err)
