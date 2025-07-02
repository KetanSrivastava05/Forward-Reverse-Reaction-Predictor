import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests

st.set_page_config(page_title="🔬 K10 Reaction Predictor", layout="wide")
st.title("🔮 K10 Forward & Reverse Reaction Predictor")
st.markdown("""
Upload a SMILES reaction or product, and this app will predict the product or reactants using **T5 transformer models** hosted on **Hugging Face**.
""")

@st.cache_resource
def load_model(repo_name):
    tokenizer = T5Tokenizer.from_pretrained(repo_name)
    model = T5ForConditionalGeneration.from_pretrained(repo_name)
    return tokenizer, model

# Load both models from Hugging Face Hub
forward_tokenizer, forward_model = load_model("K10S/forward_reaction_predictor")
reverse_tokenizer, reverse_model = load_model("K10S/reverse_reaction_predictor")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Forward Reaction (Reactants + Reagents ➝ Product)")
    forward_input = st.text_input("Enter SMILES for Reactants + Reagents (separated by '.')")
    if st.button("🚀 Predict Product"):
        if forward_input:
            input_ids = forward_tokenizer(forward_input, return_tensors="pt").input_ids
            output = forward_model.generate(input_ids, max_length=128, num_beams=5)
            predicted_smiles = forward_tokenizer.decode(output[0], skip_special_tokens=True)

            st.success("🧬 Predicted Product SMILES:")
            st.code(predicted_smiles)

            

with col2:
    st.subheader("🔄 Reverse Reaction (Product ➝ Reactants + Reagents)")
    reverse_input = st.text_input("Enter Product SMILES:")
    if st.button("🔁 Predict Reactants & Reagents"):
        if reverse_input:
            input_ids = reverse_tokenizer(reverse_input, return_tensors="pt").input_ids
            output = reverse_model.generate(input_ids, max_length=128, num_beams=5)
            predicted_reverse = reverse_tokenizer.decode(output[0], skip_special_tokens=True)

            st.success("⚗️ Predicted Reactants + Reagents SMILES:")
            st.code(predicted_reverse)

           