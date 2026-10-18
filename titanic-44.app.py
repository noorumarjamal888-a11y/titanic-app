# Run once:  python make_titanic_app.py
# Creates folder "titanic_app_files" with: app.py, requirements.txt, svc_trained_model.pkl
import os, pickle
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CATS = {'PClass': ['First', 'Second', 'Third'], 'Gender': ['Male', 'Female'], 'Sibling': ['Zero', 'One', 'Two', 'Three'], 'Embarked': ['Southampton', 'Cherbourg', 'Queenstown']}
TARGET = ["No", "Yes"]
ROWS = """
20100 11001 20000 21301 20020 01301 20000 20001 00000 10001 20120 00011 00010 11001
10000 20111 20210 20001 20000 21111 20010 01001 00010 11001 21100 21001 20000 20001
20000 21121 21200 01011 20200 01001 00110 01001 00100 21121 21100 11101 11100 21021
20010 01011 20300 21101 20000 20001 20100 11001 00300 20001 21100 11001 21300 21001
21000 21101 20010 20001 10000 01101 20320 21021 11000 21011 20000 10001 20000 00001
20000 10201 20010 21001 21000 01101 20100 11001 20000 21201 20200 21001 20000 00101
10000 21121 10000 01101 20000 11001 10000 11001 20010 00101 00100 20111 20120 20101
10100 20101
""".split()   # each row = one digit per column (position in its category list) + last digit = target (0=No, 1=Yes)

APP = r'''import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

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
'''

names = list(CATS)
X = pd.DataFrame([[int(ch) for ch in r[:-1]] for r in ROWS], columns=names)
y = [int(r[-1]) for r in ROWS]
# LabelEncoder sorts categories alphabetically, so convert to the same numbers
for n in names:
    order = sorted(CATS[n])
    X[n] = X[n].map(lambda i: order.index(CATS[n][i]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
model = svm.SVC(gamma="auto", random_state=0)
model.fit(X_train, y_train)
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 2))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_app_files")
os.makedirs(out, exist_ok=True)
pickle.dump(model, open(os.path.join(out, "svc_trained_model.pkl"), "wb"))
open(os.path.join(out, "requirements.txt"), "w").write("streamlit\nscikit-learn==" + sklearn.__version__ + "\npandas\nnumpy\n")
open(os.path.join(out, "app.py"), "w", encoding="utf-8").write(APP)
print("DONE! 3 files ban gayi hain, folder:", out)
