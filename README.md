# Powerpredict
Daily power consumption is related to the weather (rain, sunshine, temperature, etc).
Prediction of the power consumption based on the weather is relevant for energy suppliers.
This project is about creating a Machine Learning Model to predict the power consumption
according to given data.

## The Dataset:
In the given Dataset we have information about **temperature, humidity, etc.** for the five citys:
* Bedrock
* Gotham City
* New New York
* Paperopoli
* Springfield

and one column that shows the **power consumption**.


## Scores for different models:

### 1. Running on the 20 features with highest correlation to power consumption

| Name                        | Training Set Score | Test Set Score |
|-----------------------------|-------------------|---------------|
| Linear Regression           | 3422.96           | 3403.61       |
| Random Forest Regressor     | 679.13            | 2314.29       |
| Polynomial Linear Regession | 2535.72           | 3095.49       |

