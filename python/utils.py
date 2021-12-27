import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from pathlib import Path

cwd = Path(os.getcwd())
MNet_location = os.path.join(str(cwd.parent.absolute()), "asset", "data", "MNet_model.h5")
model_location = os.path.join(str(cwd.parent.absolute()), "asset", "data", "MNetDNN_model.h5")

MNet = load_model(MNet_location)
model = load_model(model_location)


def compute(file):
    img = parse_contents(file)
    if isinstance(img, str):
        return "Not an image / Error"
    else:
        X, img_resized = preprocess(img)
        print(X.shape)
        label, y_pred = prediction(X)
        return img_resized[..., ::-1], label, y_pred


def parse_contents(file):
    f = file.split("base64")
    file_type = f[0].split("/")[0].split(":")[1]
    b64_str = f[1][1:]
    if file_type != "image":
        return "You must upload an image"

    img_data = base64.b64decode(b64_str)
    img_array = np.fromstring(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def preprocess(X, target_shape=(224, 224)):
    X_resized = cv2.resize(X, target_shape)
    new_img = cv2.cvtColor(X_resized, cv2.COLOR_BGRA2BGR)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    new_img = new_img / 255
    new_img = new_img[..., ::-1]
    new_img = np.expand_dims(np.array(new_img), axis=0)
    return new_img, X_resized


def prediction(X):
    MNet_features = MNet.predict(X)
    y_pred = model.predict(MNet_features)[0][0]
    label = 1 if y_pred > .5 else 0
    return label, y_pred
