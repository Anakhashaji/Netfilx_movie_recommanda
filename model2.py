import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read the CSV file
df = pd.read_csv('NetflixOriginals.csv', encoding="latin1")

# Preprocess the 'Premiere' column
def preprocess_premiere(date_str):
    try:
        return pd.to_datetime(date_str, format='%B %d, %Y').month
    except ValueError:
        return pd.to_datetime(date_str, format='%B .%d. %Y').month

df['Premiere'] = df['Premiere'].apply(preprocess_premiere)

# Encode categorical variables
label_encoder_genre = LabelEncoder()
df['Genre'] = label_encoder_genre.fit_transform(df['Genre'])

label_encoder_language = LabelEncoder()
df['Language'] = label_encoder_language.fit_transform(df['Language'])

# Prepare the features and target
X = df[['Premiere', 'Runtime', 'IMDB Score', 'Language']]
y = df['Genre']

# Train a Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction function
def predict_genre(month, runtime, imdb_score, language):
    language_encoded = label_encoder_language.transform([language])[0]
    prediction = model.predict([[int(month), int(runtime), float(imdb_score), language_encoded]])
    genre_decoded = label_encoder_genre.inverse_transform(prediction)[0]
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    #print("Accuracy Score:", accuracy)
    
    return accuracy, genre_decoded

# Example usage of the prediction function
# Uncomment the following lines to take user input and make a prediction
#month = input("Month (1-12): ")  # Example month (e.g., 8 for August)
#runtime = input("Runtime: ")  # Example runtime in minutes
#imdb_score = input("IMDb Score: ")  # Example IMDb score
#language = input("Language: ")  # Example language
#predicted_genre = predict_genre(month, runtime, imdb_score, language)
#print(f"Predicted genre for a movie premiering in month {month} with a runtime of {runtime} minutes, IMDb score of {imdb_score}, and language {language}: {predicted_genre[1]}")
