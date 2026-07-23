import os
import gdown
import numpy as np
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MODEL_PATH = "model/eff.weights.h5"

# Replace with your actual Google Drive File ID
FILE_ID = "1QfUcessxUkYiBmXAwgswDbhbAHUtfWbO"

# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model", exist_ok=True)

# Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Class Labels
class_labels = [
    "Early_blight",
    "Late_blight",
    "healthy"
]

def build_model():
    base_model = EfficientNetB7(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="max"
    )

    base_model.trainable = False

    x = base_model.output
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model


def download_model():

    if os.path.exists(MODEL_PATH):
        print("Model already exists.")
        return

    print("=" * 60)
    print("Downloading model from Google Drive...")
    print("=" * 60)

    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model download failed.")

    print("Model downloaded successfully.")


download_model()

model = build_model()
model.load_weights(MODEL_PATH)

print("Model loaded successfully.")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = tf.keras.applications.efficientnet.preprocess_input(
        img_array
    )

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/", methods=["GET", "POST"])
def upload_predict():

    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = np.argmax(prediction)

                predicted_class = class_labels[predicted_index]

                confidence = float(
                    prediction[0][predicted_index]
                ) * 100

                return render_template(
                    "index.html",
                    prediction=predicted_class,
                    confidence=round(confidence, 2),
                    img_filename=filename
                )

            except Exception as e:
                return f"Prediction Error: {e}"

    return render_template(
        "index.html",
        prediction=None
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
