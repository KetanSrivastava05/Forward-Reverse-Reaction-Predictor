
🔬 K10 Reaction Predictor
A Streamlit web application that uses fine-tuned T5 transformer models to predict:

Forward Reactions (Reactants + Reagents ➝ Product)

Reverse Reactions (Product ➝ Reactants + Reagents)

This project leverages the power of transformers and SMILES notationfor intelligent reaction prediction.

🚀 Features
⚗️ Forward Prediction: Enter reactants and reagents in SMILES, get predicted product.

🔁 Reverse Prediction: Input a product in SMILES, get probable reactants & reagents.

🧠 Powered by fine-tuned T5-small models.

📡 Hosted on Hugging Face Hub:

K10S/forward_reaction_predictor

K10S/reverse_reaction_predictor



🧪 Dataset & Training Process
Source file: reaction.txt containing SMILES reaction strings.

Each reaction was parsed into:

Reactants > Reagents > Product

🔁 Data Preparation
Used only 40% of dataset to reduce training time.

Reactions were split into:

Forward: input = reactants.reagents, target = product

Reverse: input = product, target = reactants.reagents

⚙️ Model Training
✅ Model: t5-small

🧹 Preprocessing: Tokenization using T5Tokenizer

🛠 Framework: Hugging Face transformers, datasets, Trainer

⏱ Optimizations:

EarlyStoppingCallback

Saved checkpoints every 500 steps

Evaluated every 500 steps

Training done on Google Colab GPU (2 hours limit)

🧠 Training Script Summary

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=...,
    eval_dataset=...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()
✅ Final models saved:

forward_partial

reverse_partial

(Name with _partial to denote that whole dataset was not used for trainign due to Google Colab GPU time limit )



💻 Streamlit App
📂 Folder Structure

K10-Reaction-Predictor/
│
├── app.py                   # Streamlit app

├── requirements.txt         # All Python dependencies

├── README.md     

├── reaction.txt             #SMILES format dataset        

🧠 Model Inference
Loaded directly from Hugging Face Hub using:

model = T5ForConditionalGeneration.from_pretrained("K10S/forward_reaction_predictor")
tokenizer = T5Tokenizer.from_pretrained("K10S/forward_reaction_predictor")
🧬 How to Run Locally

git clone https://github.com/yourusername/K10-Reaction-Predictor
cd K10-Reaction-Predictor
pip install -r requirements.txt
▶️ Run the app

streamlit run app.py
