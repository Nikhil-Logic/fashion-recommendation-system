from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import uuid
import requests
import pickle

# Load embeddings.pkl and filenames.pkl from Hugging Face via HTTP
EMBEDDING_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/embeddings.pkl"
FILENAMES_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/filenames.pkl"

# Fetch the files
embedding_response = requests.get(EMBEDDING_URL)
filename_response = requests.get(FILENAMES_URL)

# Deserialize
feature_list = np.array(pickle.loads(embedding_response.content))
filenames = pickle.loads(filename_response.content)

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])
model.trainable = False

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory setup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Static mount for image files
app.mount("/images", StaticFiles(directory="images"), name="images")


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]  # skip self match


@app.post("/recommend")
async def recommend_fashion(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

        recommended_images = [f"/images/{os.path.basename(filenames[i])}" for i in indices]

        return JSONResponse(content={"recommendations": recommended_images})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run app: uvicorn fashion_app:app --reload
if __name__ == "__main__":
    uvicorn.run("fashion_app:app", host="0.0.0.0", port=8000, reload=True)
