import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open('best random forest.pkl', 'rb') as file:
    model = pickle.load(file)

# the pickle file was created using colab notebook which use sklearn 1.6.1 so we install that here and now we can do our work

def predict_charges(age, sex, bmi, children, smoker, region):
    input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region],
        'age_squared': [age**2]
    })
    prediction = model.predict(input)

    return f"Estimated Insurance Charges: ${prediction[0]:,.2f}"

inputs = [
    gr.Slider(18, 64, step=1, label="Age", value=30),
    gr.Radio(
        ["male", "female"],
        label="Gender",
        value="male"
    ),
    gr.Number(label="BMI", value=25.0),
    gr.Slider(
        0, 5, step=1,
        label="Number of Children",
        value=0
    ),
    gr.Radio(
        ["yes", "no"],
        label="Smoker",
        value="no"
    ),
    gr.Dropdown(
        ["southwest", "southeast", "northwest", "northeast"],
        label="Region",
        value="southwest"
    )
]

demo = gr.Interface(
    fn=predict_charges,
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Insurance Charges"),
    title="Medical Insurance Cost Prediction",
)

demo.launch()