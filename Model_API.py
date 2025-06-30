import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset and train the model
dataset = pd.read_csv('D:\\matches.csv')
dataset = dataset.dropna(subset=['winner', 'city'])

features = ['season', 'city', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision']
X = dataset[features]
y = dataset['winner']

# Encoding the labels
le = LabelEncoder()
y = le.fit_transform(y)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Streamlit UI
st.title("Cricket Match Winner Prediction")

# Input form for user details
season = st.selectbox('Season', ['2025/26','2024/25', '2023/24', '2022/23'])
city = st.selectbox('City', dataset['city'].unique())
venue = st.selectbox('Venue', dataset['venue'].unique())
team1 = st.selectbox('Team 1', dataset['team1'].unique())
team2 = st.selectbox('Team 2', dataset['team2'].unique())
toss_winner = st.selectbox('Toss Winner', [team1, team2])
toss_decision = st.selectbox('Toss Decision', ['field', 'bat'])

# Prepare input data for prediction
match_input = pd.DataFrame([{
    'season': season,
    'city': city,
    'venue': venue,
    'team1': team1,
    'team2': team2,
    'toss_winner': toss_winner,
    'toss_decision': toss_decision
}])

# Encoding the input data
match_input_encoded = pd.get_dummies(match_input)
match_input_encoded = match_input_encoded.reindex(columns=X_train.columns, fill_value=0)

# Predicting the winner
pred = clf.predict(match_input_encoded)
predicted_winner = le.inverse_transform(pred)

# Display the predicted winner
st.subheader(f"Predicted Winner: {predicted_winner[0]}")
