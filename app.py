import streamlit as st
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

st.title("🧠 Teachable Machine – Logistic, Random Forest & CNN")
st.write("Upload images, train models, and test predictions. Fully runs in Streamlit Cloud.")


# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    return img


# -------------------------------
# CNN Model
# -------------------------------
def build_cnn(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------------
# CLASS CREATION
# -------------------------------
st.header("📌 Step 1: Create Classes & Upload Images")

num_classes = st.number_input("How many classes?", min_value=2, step=1)

uploaded_data = {}

for i in range(num_classes):
    cname = st.text_input(f"Class {i+1} Name:")
    imgs = st.file_uploader(
        f"Upload images for {cname}",
        type=['jpg','jpeg','png'],
        accept_multiple_files=True,
        key=f"class_{i}"
    )
    uploaded_data[cname] = imgs


# -------------------------------
# MODEL TRAINING
# -------------------------------
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "CNN"])

train_btn = st.button("Train Selected Model")

if train_btn:
    X, y = [], []

    for cname, files in uploaded_data.items():
        if len(files) < 2:
            st.error(f"Class '{cname}' must have at least 2 images.")
            st.stop()

        for f in files:
            img = Image.open(f).convert("RGB")
            X.append(preprocess_image(img))
            y.append(cname)

    X = np.array(X)
    y = np.array(y)

    # Encode class names → integers
    class_names = sorted(list(uploaded_data.keys()))
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    y_idx = np.array([class_to_idx[c] for c in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42
    )

    st.write("Training model... please wait ⏳")

    # LOGISTIC REGRESSION
    if model_choice == "Logistic Regression":
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)

    # RANDOM FOREST
    elif model_choice == "Random Forest":
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        model = RandomForestClassifier(n_estimators=200)
        model.fit(X_train_flat, y_train)
        preds = model.predict(X_test_flat)

    # CNN
    else:
        model = build_cnn(num_classes)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

        model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=1)
        preds = np.argmax(model.predict(X_test), axis=1)

    # METRICS
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds, target_names=class_names)

    st.success(f"Training complete! Accuracy: {acc*100:.2f}%")

    st.write("### 📊 Confusion Matrix")
    st.dataframe(cm)

    st.write("### 📝 Classification Report")
    st.text(cr)

    # Store for predictions
    st.session_state["model"] = model
    st.session_state["type"] = model_choice
    st.session_state["classes"] = class_names


# -------------------------------
# PREDICTION
# -------------------------------
st.header("🔍 Step 3: Predict Images")

pred_imgs = st.file_uploader(
    "Upload images to predict",
    type=['jpg','jpeg','png'],
    accept_multiple_files=True,
    key="predict"
)

pred_btn = st.button("Predict Now")

if pred_btn:

    if "model" not in st.session_state:
        st.error("Train a model first!")
        st.stop()

    model = st.session_state["model"]
    model_type = st.session_state["type"]
    class_names = st.session_state["classes"]

    for f in pred_imgs:
        img = Image.open(f).convert("RGB")
        p = preprocess_image(img)

        if model_type == "CNN":
            pred = np.argmax(model.predict(np.array([p])))
        else:
            pred = model.predict(p.reshape(1, -1))[0]

        class_name = class_names[pred]

        st.image(img, width=200)
        st.success(f"Prediction: **{class_name}**")
