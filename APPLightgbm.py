import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# python -m venv smile
# smiles\scripts\activate

# Load trained model
model = joblib.load("finalized_lightgbm_model.pkl")

# Load feature columns from your dataset (excluding SMILES & target)
training_data = pd.read_csv("NaN_drop_MDM2.csv")
feature_columns = [col for col in training_data.columns if col not in ["SMILES", "pIC50"]]

# Define RDKit descriptor calculator
descriptor_names = feature_columns
calc = MolecularDescriptorCalculator(descriptor_names)

# Function to compute descriptors from SMILES
def smiles_to_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = calc.CalcDescriptors(mol)
    df = pd.DataFrame([descriptors], columns=descriptor_names)
    return df

# Streamlit UI
st.title("pIC50 Prediction Dashboard")
st.write("Upload or enter a **SMILES** to predict pIC50 using LightGBM model")

# Input box
smiles_input = st.text_input("Enter SMILES string:")

if st.button("Predict"):
    desc_df = smiles_to_descriptors(smiles_input)
    if desc_df is None:
        st.error("Invalid SMILES string!")
    else:
        # Make prediction
        prediction = model.predict(desc_df)[0]
        st.success(f"Predicted pIC50: {prediction:.3f}")
