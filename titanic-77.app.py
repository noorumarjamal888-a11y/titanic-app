import pickle
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Prediction System")
st.write("Select the passenger's details and click **Predict**.")

# Recreate the Label Encoders exactly as in the notebook (same category lists)
PCLASS = ["First", "Second", "Third"]
GENDER = ["Male", "Female"]
SIBLING = ["Zero", "One", "Two", "Three"]
EMBARKED = ["Southampton", "Cherbourg", "Queenstown"]

def make_encoder(values):
    enc = LabelEncoder()
    enc.fit(values)
    return enc

pclass_enc, gender_enc, sibling_enc, embarked_enc = map(make_encoder, [PCLASS, GENDER, SIBLING, EMBARKED])

@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

pclass = st.selectbox("Passenger class (PClass)", PCLASS)
gender = st.selectbox("Gender", GENDER)
sibling = st.selectbox("Number of siblings/spouses aboard (Sibling)", SIBLING)
embarked = st.selectbox("Port of embarkation (Embarked)", EMBARKED)

if st.button("Predict"):
    features = pd.DataFrame({
        "PClass": pclass_enc.transform([pclass]),
        "Gender": gender_enc.transform([gender]),
        "Sibling": sibling_enc.transform([sibling]),
        "Embarked": embarked_enc.transform([embarked]),
    })
    if int(model.predict(features)[0]) == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
