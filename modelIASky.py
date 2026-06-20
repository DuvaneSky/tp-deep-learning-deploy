import streamlit as st
import joblib
import numpy as np
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="TP Deep Learning", layout="centered")

st.title("TP Apprentissage profond")
st.write("Par FOZEN POKAM Franck Duvane")
st.write("Matricule: UN21TEL021IY")
st.write("Application de test des modèles entraînés")
st.write("Bank Marketing : prédiction de la souscription d'un client à un dépôt à terme, à partir de son profil (modèles Random Forest et ANN).)
st.write("Fashion MNIST : classification d'une image de vêtement parmi 10 catégories (modèle CNN).")


# Chargement des modèles et des encodeurs (mis en cache)
@st.cache_resource
def load_models():
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    ann_model = keras.models.load_model("model_ann.keras")
    cnn_model = keras.models.load_model("model_cnn.keras")
    encoders = joblib.load("label_encoders.pkl")
    return rf_model, scaler, ann_model, cnn_model, encoders

rf_model, scaler, ann_model, cnn_model, encoders = load_models()
st.success("Modèles chargés avec succès !")

tab1, tab2 = st.tabs(["Bank Marketing (RF / ANN)", "Fashion MNIST (CNN)"])

# ---------- ONGLET 1 : BANK MARKETING ----------
with tab1:
    st.header("Prédiction de souscription bancaire")
    st.write("Renseignez le profil d'un client pour prédire s'il souscrira au dépôt à terme.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=40)
        duration = st.number_input("Durée du dernier appel (secondes)", min_value=0, value=200)
        campaign = st.number_input("Nombre de contacts (campagne)", min_value=1, value=2)
        pdays = st.number_input("Jours depuis le dernier contact (999 = jamais)", value=999)
        previous = st.number_input("Nombre de contacts précédents", min_value=0, value=0)

    with col2:
        emp_var_rate = st.number_input("Taux de variation de l'emploi", value=1.1, format="%.2f")
        cons_price_idx = st.number_input("Indice des prix à la consommation", value=93.99, format="%.2f")
        cons_conf_idx = st.number_input("Indice de confiance des consommateurs", value=-36.4, format="%.2f")
        euribor3m = st.number_input("Taux Euribor 3 mois", value=4.86, format="%.2f")
        nr_employed = st.number_input("Nombre d'employés (milliers)", value=5191.0, format="%.1f")

    st.write("Profil du client :")
    col3, col4, col5 = st.columns(3)
    with col3:
        job = st.selectbox("Profession", list(encoders['job'].classes_))
        marital = st.selectbox("Statut marital", list(encoders['marital'].classes_))
    with col4:
        education = st.selectbox("Éducation", list(encoders['education'].classes_))
        default = st.selectbox("Défaut de crédit", list(encoders['default'].classes_))
    with col5:
        housing = st.selectbox("Prêt immobilier", list(encoders['housing'].classes_))
        loan = st.selectbox("Prêt personnel", list(encoders['loan'].classes_))

    contact = st.selectbox("Type de contact", list(encoders['contact'].classes_))
    month = st.selectbox("Mois", list(encoders['month'].classes_))
    day_of_week = st.selectbox("Jour de la semaine", list(encoders['day_of_week'].classes_))
    poutcome = st.selectbox("Résultat de la campagne précédente", list(encoders['poutcome'].classes_))

    model_choice = st.radio("Choisir le modèle", ["Random Forest", "ANN"])

    if st.button("Prédire"):
        # transformation des variables catégorielles avec les VRAIS encodeurs du notebook
        job_enc = encoders['job'].transform([job])[0]
        marital_enc = encoders['marital'].transform([marital])[0]
        education_enc = encoders['education'].transform([education])[0]
        default_enc = encoders['default'].transform([default])[0]
        housing_enc = encoders['housing'].transform([housing])[0]
        loan_enc = encoders['loan'].transform([loan])[0]
        contact_enc = encoders['contact'].transform([contact])[0]
        month_enc = encoders['month'].transform([month])[0]
        day_of_week_enc = encoders['day_of_week'].transform([day_of_week])[0]
        poutcome_enc = encoders['poutcome'].transform([poutcome])[0]

        # Ordre des colonnes : à adapter EXACTEMENT à l'ordre utilisé pendant l'entraînement
        features = np.array([[age, job_enc, marital_enc, education_enc, default_enc,
                               housing_enc, loan_enc, contact_enc, month_enc, day_of_week_enc,
                               duration, campaign, pdays, previous, poutcome_enc,
                               emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]])

        features_scaled = scaler.transform(features)

        if model_choice == "Random Forest":
            pred = rf_model.predict(features_scaled)[0]
            proba = rf_model.predict_proba(features_scaled)[0][1]
        else:
            proba = ann_model.predict(features_scaled)[0][0]
            pred = 1 if proba > 0.5 else 0

        if pred == 1:
            st.success(f"Le client va probablement SOUSCRIRE (probabilité : {proba:.2%})")
        else:
            st.info(f"Le client ne va probablement PAS souscrire (probabilité de souscription : {proba:.2%})")

# ---------- ONGLET 2 : FASHION MNIST ----------
with tab2:
    st.header("Classification d'image (Fashion MNIST)")
    st.write("Importez une image de vêtement (28x28 pixels, niveaux de gris de préférence).")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image_resized = image.resize((28, 28))
        st.image(image_resized, caption="Image redimensionnée (28x28)", width=150)

        img_array = np.array(image_resized).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        pred = cnn_model.predict(img_array)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred)

        st.success(f"Classe prédite : **{class_names[predicted_class]}** (confiance : {confidence:.2%})")
