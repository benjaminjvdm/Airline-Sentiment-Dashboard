# Sentiment Analysis of Tweets about US Airlines

This repository contains a Streamlit dashboard used to analyze the sentiments of tweets about US airlines. The application displays visualizations and breakdowns of tweet sentiment, time of day, airline choice, and word cloud for specific sentiments.

## Table of Contents
- [Libraries Used](#libraries-used)
- [How to Run the Code](#how-to-run-the-code)
- [Data](#data)
- [Functionality](#functionality)
- [Contact](#contact)

## Libraries Used
The following libraries were used in this project:
* Streamlit - used to build the web app 
* Pandas - used to handle data
* Numpy - used for numerical computing
* Plotly Express - used to create interactive visualizations
* Plotly Subplots - used to create subplots
* Plotly Graph Objects - used to create additional plotly objects
* Wordcloud - used to generate word clouds
* Matplotlib - used for plotting visuals

## How to Run the Code
1. Install the necessary libraries using the following command:

```python
pip install -r requirements.txt
```

2. Run the `app.py` file using the following command:

```python
streamlit run app.py
```

3. The web app should be up and running on your default browser.

Note: The data used in this project is available in the `Tweets.csv` file.

## Data
The data used in this project is available in the `Tweets.csv` file.

## Functionality
The app allows users to do the following:
* Analyze the number of tweets by sentiment using bar or pie charts
* Explore when and where users are tweeting from using a map
* View the total number of tweets for each airline using bar or pie charts
* Breakdown tweets by airline and sentiment using bar or pie charts
* Display a word cloud for a specific sentiment
