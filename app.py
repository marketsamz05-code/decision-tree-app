import streamlit as st
import joblib
import pandas as pd

st.title("My Decision Tree Model")
st.write("Enter the four Iris flower measurements.")

model = joblib.load("decision_tree_model.pkl")

values = [
    st.number_input("Sepal length", value=5.1),
    st.number_input("Sepal width", value=3.5),
    st.number_input("Petal length", value=1.4),
    st.number_input("Petal width", value=0.2)
]

if st.button("Predict"):
    X = pd.DataFrame(
        [values],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )
    result = int(model.predict(X)[0])
    st.success(["Setosa", "Versicolor", "Virginica"][result])
