# Game Recommender

A machine learning-based web application for recommending video games and exploring video game review analytics. Built with FastAPI, Streamlit, and data from IGN and Metacritic, this project allows users to discover game trends, analyze genre and platform popularity, and receive personalized game recommendations.

## Features

- **Game Recommendation System:**  
  Enter a game title and get recommendations for similar games based on genre, platform, release year, and review score.

- **Review Analytics Dashboard:**  
  Visualize trends in video game reviews over years and months, explore genre popularity, and analyze the distribution of games across platforms.

- **Data Preprocessing and Aggregation:**  
  Combines and cleans data from IGN and Metacritic, processes genres and platforms, and prepares datasets for analysis and model training.

## Technologies Used

- **Python** (Jupyter Notebook, pandas, scikit-learn)
- **FastAPI** for backend API serving recommendation endpoints
- **Streamlit** for interactive dashboards and visualization
- **Plotly** for data visualization in Streamlit
- **Requests** for API calls within the dashboard

## Project Structure

- `main.py`  
  FastAPI backend providing endpoints for analytics and recommendations.

- `streamlit_app.py`  
  Streamlit dashboard visualizing review trends and serving as the frontend for the recommender.

- `data_preprocessing.ipynb`  
  Jupyter Notebook for cleaning and merging raw data from CSV files.

- `merged_data.csv`, `IGN games from best to worst.csv`, `cleaned_data.csv`, `metacritic.csv`  
  CSV files containing raw and processed video game review data.

- `Final Project FGA.pptx`  
  Project presentation.

## Setup & Usage

### 1. Data Preparation

- Ensure all CSV files (`IGN games from best to worst.csv`, `cleaned_data.csv`, `metacritic.csv`) are present.
- Run `data_preprocessing.ipynb` to clean, merge, and generate `merged_data.csv`.

### 2. Start the Backend API

```bash
pip install fastapi uvicorn pandas scikit-learn
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### 3. Start the Streamlit Dashboard

```bash
pip install streamlit plotly requests pandas
streamlit run streamlit_app.py
```
The dashboard will run at `http://localhost:8501` and connect to the FastAPI backend.

## API Endpoints

- `/reviews-by-year` – Number of reviews per year
- `/reviews-by-month` – Number of reviews by month (2014–2024)
- `/genre-popularity` – Average score and count for top genres
- `/platform-distribution` – Platform distribution by release year
- `/recommend` – Game recommendations (parameters: `game_title`, `n`)

## Example: Requesting Recommendations

Send a GET request to:
```
http://localhost:8000/recommend?game_title=YourGameTitle&n=5
```
Returns a list of recommended games with their platform, genre, and score.
