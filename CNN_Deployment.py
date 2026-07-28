import uvicorn
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential,load_model
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
loaded_model = tf.keras.models.load_model(r"C:\Users\Abdul\OneDrive\Documents\Ultimate AI Mastery BootCamp\Deep Learning\Session 27\classifier.keras")
app = FastAPI()

@app.get('/')
def index():
    return {'Deployment': 'Hello and Welcome to AI Engineering Deployment Project'}


@app.post("/predict")
async def predict1(
    file: UploadFile = File(...)):
    

    image = await file.read()
    image = Image.open(BytesIO(image))
    image = image.convert('RGB')

    pic = np.array(image)

    pic = pic / 255
    pic = np.expand_dims(pic, axis=0)
    predicted = loaded_model.predict(pic)
    prediction = predicted[0]
    if(prediction < 0.5):
        output = 'Cat'
    else:
        output = 'Dog'

    return {"Prediction": output}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5002)