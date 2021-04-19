import pandas as pd

from lib.data_processing import standard_scaler, simple_imputer, ordinal_encoder
from src.fit_models import fit_models
from src.visualise import visualise

pd.options.mode.chained_assignment = None

# import data and test
stars_data = pd.read_csv("stars.csv")

# test dataset
print(stars_data.info())
print()  # newline

numerical_features = ['Temperature', 'Luminosity', 'Radius', 'Magnitude']
categorical_features = ['Color', 'Spectral_Class']


# correct bad names
def switch_names(text):
    return {
        'Blue-white': 'Blue-White',
        'Blue White': 'Blue-White',
        'Blue white': 'Blue-White',
        'yellow-white': 'White-Yellow',
        'Yellowish White': 'White-Yellow',
        'white': 'White',
        'yellowish': 'Yellowish',
        'Pale yellow orange': 'Yellow-Orange'
    }.get(text, text)


for i in range(len(stars_data)):
    text = stars_data.get("Color")[i]
    stars_data.get("Color")[i] = switch_names(text)

# split targets and features
X = stars_data.drop(columns="Type")
y = stars_data["Type"]

### VISUALISATION ###

visualise(stars_data, X, y)

### ENCODING ###

X[numerical_features] = simple_imputer(X[numerical_features], strategy="median")
X[numerical_features] = standard_scaler(X[numerical_features])

X[categorical_features] = simple_imputer(X[categorical_features], strategy="constant")
X[categorical_features] = ordinal_encoder(X[categorical_features])

### FIT MODELS ###

fit_models(X, y)
