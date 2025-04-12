from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .recommender import get_event_recommendations, get_location_recommendations
from models.models_setup import load_data_and_models
import pandas as pd
import os
import sys

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Use relative path to load the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), "locations.csv")
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(r"C:\Users\Kushal\Desktop\locations.csv")
    print("Data loaded successfully. Shape:", df.shape)
    
    # Load models with detailed debugging
    print("\n=== LOADING MODELS ===")
    models = load_data_and_models()
    
    print(f"Models type: {type(models)}")
    if hasattr(models, '__len__'):
        print(f"Number of items returned: {len(models)}")
        print(f"First 3 items types: {[type(x) for x in models[:3]]}")
    else:
        print("Returned object has no length attribute")
    
    # Unpack models with flexible handling
    if isinstance(models, (tuple, list)) and len(models) >= 4:
        event_vectorizer, event_model, location_model, tfidf_matrix = models[:4]
    elif isinstance(models, dict):
        event_vectorizer = models['event_vectorizer']
        event_model = models['event_model']
        location_model = models['location_model']
        tfidf_matrix = models['tfidf_matrix']
    else:
        raise ValueError(f"Unexpected return format from load_data_and_models()")
    
    print("\n=== MODEL TYPES ===")
    print(f"event_vectorizer: {type(event_vectorizer)}")
    print(f"event_model: {type(event_model)}")
    print(f"location_model: {type(location_model)}")
    print(f"tfidf_matrix: {type(tfidf_matrix)}")
    
    print("\nInitialization completed successfully!")

except FileNotFoundError as e:
    print(f"\nERROR: File not found: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"\nERROR during initialization: {str(e)}", file=sys.stderr)
    print("\nDEBUG INFO:")
    if 'models' in locals():
        print(f"Models object: {models}")
        print(f"Type: {type(models)}")
        if hasattr(models, '__len__'):
            print(f"Length: {len(models)}")
            print(f"Items: {models}")
    sys.exit(1)

@app.get("/recommend/events")
async def recommend_events(query: str = Query(..., min_length=2)):
    try:
        return get_event_recommendations(
            query, 
            df, 
            event_vectorizer, 
            event_model, 
            tfidf_matrix
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/locations")
async def recommend_locations(location: str = Query(..., min_length=2)):
    try:
        return get_location_recommendations(location, df, location_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
