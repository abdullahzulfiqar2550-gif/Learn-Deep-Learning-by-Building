import uvicorn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, File ,UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd

app = FastAPI()

@app.get("/")
def index():
    return {"message": "Hello, and Welcome to AI Engineering "}


@app.post("/predict")
async def predict1(
    File: UploadFile = File(...)):
    train = pd.read_csv(r"C:\\Users\\Abdul\\OneDrive\\Documents\\Ultimate AI Mastery BootCamp\\Deep Learning\\Session 25&26\\mnist_train.csv")
    test = pd.read_csv(r"C:\\Users\\Abdul\\OneDrive\\Documents\\Ultimate AI Mastery BootCamp\\Deep Learning\\Session 25&26\\mnist_test.csv")

    y_train = train.iloc[:, 0]
    X_train = train.iloc[:, 1:]
    y_test = test.iloc[:, 0]
    X_test = test.iloc[:, 1:]

    X_train = X_train / 255
    X_test = X_test / 255
    X_train_flattened = X_train.values.reshape(len(X_train), 28*28)
    X_test_flattened = X_test.values.reshape(len(X_test), 28*28)


    model = keras.Sequential([
    keras.layers.Dense(300, input_dim=(784), activation='relu'),
    keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
    y_train = y_train.astype(str).astype(int)
    print(y_train)
    model.fit(X_train_flattened, y_train, epochs=3)

    image = await File.read()
    image = Image.open(BytesIO(image))
    image = image.convert('L')

    pic = np.array(image)

    pic = pic / 255
    pic_flattened = pic.reshape(28*28)
    pic_flattened = np.expand_dims(pic_flattened, axis=0)
    predicted = model.predict(pic_flattened)
    prediction = np.argmax(predicted[0])
    if(prediction == 0):
        output = 'zero'
    if(prediction == 1):
        output = 'one'
    if(prediction == 2):
        output = 'two'
    if(prediction == 3):
        output = 'three'
    if(prediction == 4):
        output = 'four'
    if(prediction == 5):
        output = 'five'
    if(prediction == 6):
        output = 'six'
    if(prediction == 7):
        output = 'seven'
    if(prediction == 8):
        output = 'eight'
    if(prediction == 9):
        output = 'nine'

    return {"Prediction": output}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.4', port=5000)