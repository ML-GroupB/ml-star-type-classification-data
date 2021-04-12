import pandas as pd
import pickle

from lib.data_processing import standard_scaler, simple_imputer, ordinal_encoder
from src.fit_models import fit_models
from visualize import visualise

pd.options.mode.chained_assignment = None

# import data and test
stars_data = pd.read_csv("../Stars.csv")

# test dataset
print(stars_data.info())
print()  # newline

# split targets and features
X = stars_data.drop(columns="Type")
y = stars_data["Type"]

numerical_features = ['Temperature', 'Luminosity', 'Radius', 'Magnitude']
categorical_features = ['Color', 'Spectral_Class']

# correct bad names
stars_data.Color.loc[stars_data.Color == 'Blue-white'] = 'Blue-White'
stars_data.Color.loc[stars_data.Color == 'Blue White'] = 'Blue-White'
stars_data.Color.loc[stars_data.Color == 'Blue white'] = 'Blue-White'
stars_data.Color.loc[stars_data.Color == 'yellow-white'] = 'White-Yellow'
stars_data.Color.loc[stars_data.Color == 'Yellowish White'] = 'White-Yellow'
stars_data.Color.loc[stars_data.Color == 'white'] = 'White'
stars_data.Color.loc[stars_data.Color == 'yellowish'] = 'Yellowish'

### VISUALISATION ###

# visualise(stars_data, X, y)

### ENCODING ###

X[numerical_features] = simple_imputer(X[numerical_features], strategy="median")
X[numerical_features] = standard_scaler(X[numerical_features])

X[categorical_features] = simple_imputer(X[categorical_features], strategy="constant")
X[categorical_features] = ordinal_encoder(X[categorical_features])

### FIT MODELS ###

fit_models(X, y)

