import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the CSV file
df = pd.read_csv('NetflixOriginals.csv', encoding="latin1")

# Parse the 'Premiere' column
df['Premiere'] = pd.to_datetime(df['Premiere'], format='%B .%d. %Y').dt.month

# Encode categorical variables
label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])

# Prepare the features and target
X = df[['Genre', 'Premiere', 'Runtime']]
y = df['IMDB Score']

# Initialize and train models
rf_model = RandomForestRegressor()
gb_model = GradientBoostingRegressor()
svr_model = SVR()

rf_model.fit(X, y)
gb_model.fit(X, y)
svr_model.fit(X, y)

ensemble_model = VotingRegressor(estimators=[
    ('rf', rf_model),
    ('gb', gb_model),
    ('svr', svr_model)
])
ensemble_model.fit(X, y)

# Prediction functions
def predict_rf_imdb_score(genre, month, runtime):
    genre_encoded = label_encoder.transform([genre])[0]
    prediction = rf_model.predict([[genre_encoded, month, runtime]])
    y_pred = rf_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return rmse, mae, prediction[0]

def predict_gb_imdb_score(genre, month, runtime):
    genre_encoded = label_encoder.transform([genre])[0]
    prediction = gb_model.predict([[genre_encoded, month, runtime]])
    y_pred = gb_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return rmse, mae, prediction[0]

def predict_svr_imdb_score(genre, month, runtime):
    genre_encoded = label_encoder.transform([genre])[0]
    prediction = svr_model.predict([[genre_encoded, month, runtime]])
    y_pred = svr_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return rmse, mae, prediction[0]

def predict_ensemble_imdb_score(genre, month, runtime):
    genre_encoded = label_encoder.transform([genre])[0]
    prediction = ensemble_model.predict([[genre_encoded, month, runtime]])
    y_pred = ensemble_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return rmse, mae, prediction[0]

# Function to return unique genres and months
def get_unique_genres_and_months():
    genres = label_encoder.inverse_transform(df['Genre'].unique())
    months = df['Premiere'].unique()
    return genres, months
