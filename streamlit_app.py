import calendar
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.header('IGN Video Game Review EDA')

# Reviews by Year
st.subheader('Review Released Over the Years')
response = requests.get(f"{API_URL}/reviews-by-year")
df_year = pd.DataFrame(response.json())
st.bar_chart(
    df_year.set_index('release_year')['title'], 
    x_label='Year', 
    y_label='number of reviews', 
    use_container_width=False, 
    height=500, width=800,
    color='title'
)

# Reviews by Month
st.subheader("Reviews Released by Month (2014-2024)")
response = requests.get(f"{API_URL}/reviews-by-month")

df_month = pd.DataFrame(response.json(), columns=['month', 'releases'])
df_month['month_name'] = df_month['month'].apply(lambda x: calendar.month_name[x])
month_order = [calendar.month_name[i] for i in range(1, 13)]
df_month['month_name'] = pd.Categorical(df_month['month_name'], categories=month_order, ordered=True)
df_month = df_month.sort_values('month_name')

color_palette = ['#00429d', '#2251a2', '#3460a7', '#436fab', '#517eaf', '#5e8eb4', '#6d9db8', '#7cadbb', '#8dbcbf', '#a1cbc1', '#bad9c4', '#e2e2c4']
df_month['color'] = color_palette

st.bar_chart(
    df_month, 
    x='month_name', y='releases', 
    width=800, height=500,
    use_container_width=False,
    color='color'
)

# Genre Popularity
st.subheader('Genre Popularity')
response = requests.get(f"{API_URL}/genre-popularity")
data = response.json()

df_genre = pd.DataFrame(data)
fig = px.scatter(df_genre, x='genre', y='average_score', size='num_games', text='num_games', color='num_games')
fig.update_traces(textposition='top center', marker=dict(size=df_genre['num_games'] * 5))
fig.update_layout(
    xaxis_title='Genre', yaxis_title='Average Score', 
    title='Genre Popularity',
    height=500,width=800
)
st.plotly_chart(fig, use_container_width=False)

# Platform Distribution
st.subheader('Platform Distribution by Year')
response = requests.get(f"{API_URL}/platform-distribution")
df_platform = pd.DataFrame(response.json())

st.line_chart(
    df_platform.set_index('release_year'), 
    x_label='Year',
    y_label='Number of Games',
    use_container_width=False, 
    height=500, width=800
    )

# Recommendation
def recommend_games(game_title):
    response = requests.get(f"{API_URL}/recommend?game_title={game_title}")
    recommendations = response.json()
    return recommendations

st.title("Machine Learning Model")
st.subheader("Game Recommender")

game_title = st.text_input("Enter a game title:")

if game_title:
    recommendations = recommend_games(game_title)
    st.write("Recommended Games:")
    for rec in recommendations:
        st.write(f"- {rec['title']} ({rec['platform']}) - genre: {rec['genre']} - score: {rec['score']}")