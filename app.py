import model23
from model2 import predict_genre 
from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import base64
import io


app = Flask(__name__)
df = pd.read_csv('NetflixOriginals.csv', encoding='latin1')
df['Premiere'] = pd.to_datetime(df['Premiere'], format='%B .%d. %Y')
df['Premiere_Month'] = df['Premiere'].dt.strftime('%B')
d1 = pd.read_csv('NetflixOriginals.csv', encoding='latin1')


# Function to generate a pie chart using Plotly
def generate_pie_chart(data, column_name):
    counts = data[column_name].value_counts().head(10)
    fig = px.pie(counts, values=counts.values, names=counts.index, title=f'Distribution of {column_name}')
    json_data = pio.to_json(fig)
    return json_data  # Return JSON-formatted string

# Function to generate a pie chart using Plotly
def generate_pie_chart2(data, column_name):
    counts = data[column_name].value_counts().head(10)
    fig = px.pie(counts, values=counts.values, names=counts.index, title=f'Distribution of {column_name}')
    json_data = pio.to_json(fig)
    return json_data

# Function to generate a bar chart using Plotly
def generate_bar_chart(data, top_n=10):
    genre_rating_mean = data.groupby('Genre')['IMDB Score'].mean().sort_values(ascending=False).head(top_n)
    fig = px.bar(genre_rating_mean, x=genre_rating_mean.index, y=genre_rating_mean.values, title='Top 10 Genres by Average Rating')
    json_data = pio.to_json(fig)
    return json_data  # Return JSON-formatted string

# Function to generate a custom chart using Plotly
def generate_custom_chart(data, x_column, y_column, num_rows):
    df_subset = data.head(num_rows)
    fig = px.bar(df_subset, x=x_column, y=y_column, title=f'Custom Chart: {y_column} vs {x_column}')
    json_data = pio.to_json(fig)
    return json_data

# Route to handle form submission and generate chart dynamically
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    data = request.json  # Use request object to access JSON data
    x_column = data['xAxisColumn']
    y_column = data['yAxisColumn']
    num_rows = int(data['numRows'])

    chart_data = generate_custom_chart(df, x_column, y_column, num_rows)
    
    return jsonify(chart_data)  # Return JSON response
# Function to generate a bar chart and return it as base64 encoded image

# Inside your dashboard route
@app.route('/')
def dashboard():
    language_pie_chart = generate_pie_chart(df, 'Language')
    Premiere_pie_chart = generate_pie_chart2(df, 'Premiere_Month')
    bar_chart = generate_bar_chart(df)
    column_names = df.columns.tolist()

    df['Year'] = pd.to_datetime(df['Premiere'].replace('.', ' ')).dt.year

    # Group by Year and calculate the mean IMDB score
    yearly_trend = df.groupby('Year')['IMDB Score'].mean().reset_index()

    # Create a Plotly figure
    fig = px.line(yearly_trend, x='Year', y='IMDB Score', markers=True, title='Yearly Trend of IMDB Scores')
    fig.update_layout(xaxis_title='Year', yaxis_title='Average IMDB Score')

    # Convert the Plotly figure to JSON
    graphJSON = fig.to_json()

    return render_template('netflixdashv2.html', plot_json=graphJSON, csv_data=d1, bar_chart=bar_chart, Premiere_pie_chart=Premiere_pie_chart, language_pie_chart=language_pie_chart, column_names=column_names)

@app.route('/recom', methods=['GET', 'POST'])
def index():
    df = pd.read_csv('NetflixOriginals.csv')
    search_title = None
    results = None
    if request.method == 'POST':
        search_title = request.form.get('title')
        if search_title:
            results = df[df['Title'].str.contains(search_title, case=False, na=False)]
            results = results.to_dict(orient='records')

    # Get unique values for genres and languages
    unique_genres = df['Genre'].unique()
    unique_languages = df['Language'].unique()

    if request.method == 'POST':
        # Get the selected genre and language from the form
        selected_genre = request.form.get('genre', '')
        selected_language = request.form.get('language', '')

        # Filter the data based on the selected genre and language
        filtered_data = df
        if selected_genre:
            filtered_data = filtered_data[filtered_data['Genre'] == selected_genre]
        if selected_language:
            filtered_data = filtered_data[filtered_data['Language'] == selected_language]

        if filtered_data.empty:
            # No movies found
            no_movies_message = "No movies found for the selected genre and language."
            return render_template(
                'recom.html',
                unique_genres=unique_genres,
                unique_languages=unique_languages,
                no_movies_message=no_movies_message,
                selected_genre=selected_genre,
                selected_language=selected_language,
                csv_data=d1
            )
        else:
            # Find the row with the highest rating within the filtered data
            row_with_highest_rating = filtered_data.loc[filtered_data['IMDB Score'].idxmax()]

            # Extract the title from the row with the highest rating
            title_with_highest_rating = row_with_highest_rating['Title']

            # Get the top ten movies for the selected genre and language
            top_ten_movies = filtered_data.sort_values(by='IMDB Score', ascending=False).head(10)['Title'].tolist()

            # Pass the data to the template
            return render_template(
                'recom.html',
                unique_genres=unique_genres,
                unique_languages=unique_languages,
                title_with_highest_rating=title_with_highest_rating,
                top_ten_movies=top_ten_movies,
                selected_genre=selected_genre,
                selected_language=selected_language,
                csv_data=d1,
                results=results
            )

    else:
        # Render the initial page with the genre and language list
        return render_template('recom.html', results=results, unique_genres=unique_genres, unique_languages=unique_languages, csv_data=d1)

# Route for predictions page
@app.route('/predic', methods=['GET', 'POST'])
def predictions():
    genres, months = model23.get_unique_genres_and_months()
    predicted_score = None
    rmse = None
    mae = None
    
    if request.method == 'POST':
        genre = request.form.get('genre', '')
        month = int(request.form.get('month', ''))
        runtime = int(request.form.get('runtime', 0))
        selected_model = request.form.get('model', '')
        
        if selected_model == 'rf':
            rmse, mae, predicted_score = model23.predict_rf_imdb_score(genre, month, runtime)
        elif selected_model == 'gb':
            rmse, mae, predicted_score = model23.predict_gb_imdb_score(genre, month, runtime)
        elif selected_model == 'svr':
            rmse, mae, predicted_score = model23.predict_svr_imdb_score(genre, month, runtime)
        elif selected_model == 'ensemble':
            rmse, mae, predicted_score = model23.predict_ensemble_imdb_score(genre, month, runtime)
        elif selected_model == 'all':
            # Predict with all models and collect results
            results = []
            models = ['rf', 'gb', 'svr', 'ensemble']
            for model in models:
                rmse, mae, predicted_score = getattr(model23, f'predict_{model}_imdb_score')(genre, month, runtime)
                results.append((model, rmse, mae, predicted_score))
                # Generate chart
            
            
            return render_template('predic.html',rmse=rmse, mae=mae, results=results, genres=genres, months=months, selected_model='all', csv_data=d1)
        
        return render_template('predic.html', rmse=rmse,mae=mae, predicted_score=predicted_score, genres=genres, months=months, selected_model=selected_model, csv_data=d1)
    
    return render_template('predic.html',rmse=rmse, mae=mae, genres=genres, months=months, selected_model=None, csv_data=d1)


@app.route('/genre' , methods=['GET','POST'])
def genre():
    languages = df['Language'].unique()
    

    if request.method == 'POST': 
        runtime = int(request.form.get('runtime', 0))
        month = int(request.form.get('month', ''))
        imdb_score =float( request.form.get('imdb_score', 0))
        language = request.form.get('language', '')
        
    

        accuracy,predicted_genre = predict_genre(month, runtime, imdb_score, language)

        return render_template('genre.html',accuracy=accuracy,predicted_genre=predicted_genre,imdb_score=imdb_score,languages=languages, csv_data=d1)
    else:
        return render_template('genre.html',languages=languages, csv_data=d1)

    
    

        


if __name__ == '__main__':
    app.run(debug=True)
