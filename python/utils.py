import base64
import numpy as np
import plotly.express as px
import cv2
from tensorflow.keras.models import load_model
from keras.applications.mobilenet import preprocess_input
import os
from pathlib import Path
from database import insert_table
import pickle

cwd = Path(os.getcwd())
MNet_location = os.path.join(str(cwd.parent.absolute()), "asset", "data", "models", "MNet.h5")


MNet = load_model(MNet_location)

headers = {"0": ["I don't see any tasty banana here", "There is no banana on this image", "I can't found any banana"],
           "1": ["I think your image contains bananas", "What a delicious banana", "I see.. mhh.. a banana here"]}


def compute(file):
    img = parse_contents(file)
    if isinstance(img, str):
        return "Not an image / Error"
    else:
        X, img_resized = preprocess(img)
        y_pred = prediction(X)
        insert_table(int(y_pred), True, pickle.dumps(img_resized))
        return img_resized, y_pred


def parse_contents(file):
    f = file.split("base64")
    file_type = f[0].split("/")[0].split(":")[1]
    extension = f[0].split("/")[1][:-1]
    b64_str = f[1][1:]

    if file_type != "image" or extension not in ["jpg", "jpeg", "png", "webp"]:
        return "You must upload an image"

    img_data = base64.b64decode(b64_str)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def preprocess(X, target_shape=(224, 224)):
    X_resized = cv2.resize(X, target_shape)
    X = preprocess_input(X_resized)
    X = np.expand_dims(np.array(X), axis=0)
    return X, np.array(X_resized)[..., ::-1]


def prediction(X):
    y_pred = np.argmax(MNet.predict(X), axis=1)[0]
    return y_pred


def get_fig(X):
    fig = px.imshow(X).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout({"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",
                       "margin": {"l": 0, "r": 0, "b": 0, "t": 0}})
    return fig
