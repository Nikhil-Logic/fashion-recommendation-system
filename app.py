from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import pickle
import requests

# ========== Constants ==========
EMBEDDING_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/embeddings.pkl"
FILENAMES_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/filenames.pkl"
HUGGINGFACE_IMAGE_BASE_URL = "https://huggingface.co/datasets/TheNikhil/fashion-product-dataset/resolve/main/images/"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ========== Fetch Embeddings & Filenames ==========
embedding_response = requests.get(EMBEDDING_URL)
filename_response = requests.get(FILENAMES_URL)

if embedding_response.status_code != 200 or filename_response.status_code != 200:
    raise RuntimeError("Failed to fetch embeddings or filenames from Hugging Face")

feature_list = np.array(pickle.loads(embedding_response.content))
filenames = pickle.loads(filename_response.content)

# ========== Model Setup ==========
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])
model.trainable = False

# ========== App Initialization ==========
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Size Limit Middleware (5MB)
class FileSizeLimiterMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if "content-length" in request.headers:
            size = int(request.headers["content-length"])
            if size > 5 * 1024 * 1024:  # 5 MB
                return JSONResponse(status_code=413, content={"error": "File too large"})
        return await call_next(request)

app.add_middleware(FileSizeLimiterMiddleware)

# ========== Health Check ==========
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ========== Feature Extraction ==========
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ========== Recommendation ==========
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]  # skip self

# ========== URL Mapping Based on Ranges ==========
def get_image_url(image_filename):
    try:
        file_id = int(os.path.splitext(os.path.basename(image_filename))[0])
    except ValueError:
        return HUGGINGFACE_IMAGE_BASE_URL + image_filename

    if file_id <= 13379:
        subfolder = "images1"
    elif file_id <= 25878:
        subfolder = "images2"
    elif file_id <= 39670:
        subfolder = "images3"
    elif file_id <= 60000:
        subfolder = "images4"
    else:
        subfolder = "images5"

    return f"{HUGGINGFACE_IMAGE_BASE_URL}{subfolder}/{image_filename}"

# ========== Recommend Endpoint ==========
@app.post("/recommend")
async def recommend_fashion(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extract features and recommend
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

        # Clean up uploaded file
        os.remove(file_path)

        # Create image URLs from Hugging Face
        recommended_images = [
            get_image_url(os.path.basename(filenames[i]))
            for i in indices
        ]

        return JSONResponse(content={"recommendations": recommended_images})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ========== Optional: Local Run ==========
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)