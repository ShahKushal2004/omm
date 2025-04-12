from rapidfuzz import process
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Event Recommendation Function
def get_event_recommendations(query, df, event_vectorizer, event_model, tfidf_matrix):
    matches = process.extract(query, df['Event Name'].astype(str), limit=5)
    best_match = matches[0][0] if matches and matches[0][1] > 50 else None
    if not best_match:
        return None

    idx = df[df['Event Name'] == best_match].index[0]
    distances, indices = event_model.kneighbors(tfidf_matrix[idx], n_neighbors=5)
    return df.iloc[indices[0]]

# Location Recommendation Function
def get_location_recommendations(location, df, location_model):
    # Try exact match first
    location_events = df[df['Location'].str.lower() == location.lower()]
    if not location_events.empty:
        return location_events.head(5)
    
    # Find nearest locations
    try:
        loc_data = df[df['Location'].str.lower() == location.lower()].iloc[0]
        distances, indices = location_model.kneighbors(
            np.radians([[loc_data["Latitude"], loc_data["Longitude"]]]), 
            n_neighbors=5
        )
        return df.iloc[indices[0]]
    except:
        return df[df['Location'].str.contains(location, case=False)].head(5)
