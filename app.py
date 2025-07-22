from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import requests
import pickle
import io
import os
from PIL import Image
import uvicorn
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# ========== Constants ==========
EMBEDDINGS_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/embeddings.pkl"
FILENAMES_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/filenames.pkl"
CSV_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/styles.csv"
CLOUDINARY_URL = "https://res.cloudinary.com/drk0sip9z/image/upload/"

# ========== FastAPI Setup ==========
app = FastAPI(title="Fashion Recommendation API")

# CORS Middleware (optional for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load Pickle & CSV ==========
def load_pickle_from_hf(url):
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

def load_csv_from_hf(url):
    response = requests.get(url)
    return pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')

feature_list = np.array(load_pickle_from_hf(EMBEDDINGS_URL))
filenames = load_pickle_from_hf(FILENAMES_URL)
styles_df = load_csv_from_hf(CSV_URL)

# ========== Load Model ==========
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([base_model, GlobalMaxPooling2D()])
    model.trainable = False
    return model

model = load_model()

# ========== Helper Functions ==========
def feature_extraction(img: Image.Image, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized = result / norm(result)
    return normalized

def recommend(features, feature_list, n=5):
    neighbors = NearestNeighbors(n_neighbors=n + 1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]  # Skip the first one (it’s the same image)

def get_metadata(image_filename):
    try:
        image_id = int(os.path.splitext(image_filename)[0])
        row = styles_df[styles_df['id'] == image_id]
        if not row.empty:
            return {
                "gender": row["gender"].values[0],
                "articleType": row["articleType"].values[0],
                "baseColour": row["baseColour"].values[0],
                "usage": row["usage"].values[0] if "usage" in row.columns else None
            }
    except:
        pass
    return {}

# ========== API Route ==========
@app.post("/recommend")
async def recommend_fashion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        features = feature_extraction(img, model)
        indices = recommend(features, feature_list)

        results = []
        for idx in indices:
            filename = os.path.basename(filenames[idx])
            image_url = CLOUDINARY_URL + filename
            metadata = get_metadata(filename)
            results.append({
                "filename": filename,
                "image_url": image_url,
                "metadata": metadata
            })

        return {"recommendations": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
