import streamlit as st
import pandas as pd
import joblib
import tempfile
import shutil
import os
import math
from padelpy import from_smiles
from rdkit import Chem
import base64

# ------------------------------
# CONFIG - Update paths
# ------------------------------
MODEL_PATHS = {
    "Model 1": "lightgbm.pkl",
    "Model 2": "rf.pkl"
}
FEATURE_PATHS = {
    "Model 1": "newm1_aligned_376.csv",
    "Model 2": "rf_ready_newm2.csv"
}
DATASET_PATH = "main.csv"  # your main dataset file

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="MDM2 pIC50 Prediction", layout="wide")

# ------------------------------
# Custom CSS (Theme)
# ------------------------------
st.markdown("""
<style>
/* Global font & background */
.stApp {
    background: #ffffff;
    color: #1a1a1a;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 10px;
    font-family: 'zimula'; 
    font-weight: bold;
    margin-bottom: 0px;
}

/* Tabs */
.stTabs [role="tablist"] {
    gap: 0px;
    display: flex;
    justify-content: center;
    border-bottom: 2px solid #006666; /* matches color */
}

/* ✅ Correct selector for tab text */
.stTabs [role="tab"] span {
    font-size: 70px !important;     /* increase font size */
    font-weight: 600 !important;
    color: #006666 !important;
    font-family: 'Segoe UI', sans-serif !important;
    font-weight: bold !important;
}

.stTabs [role="tab"] {
    padding: 40px 82px !important;  /* keep your tab size */
    background-color: #e6ffff !important;
    border-radius: 0px 0px 0 0 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #006666 !important;
    color: white !important;
}

.stTabs [aria-selected="true"] {
    background-color: #006666 !important;
    color: white !important;
    border-radius: 10px 10px 0 0;
    box-shadow: 0px -4px 12px rgba(0,0,0,0.25);
    transform: translateY(-4px) scale(1.05); /* lift + zoom */
    transition: all 0.3s ease-in-out;
}

/* Buttons */
.stButton>button {
    background-color: #006666;
    color: white;
    border-radius: 8px;
    font-size: 100px;
    font-weight: bold;
    padding: 6px 20px;
}
.stButton>button:hover {
    background-color: #002244;
}

/* Small example text */
.example-text {
    font-size: 20px;
    color: #444;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Converter Functions
# ------------------------------
def pIC50_to_IC50(pIC50):
    return 10 ** (-pIC50)

def IC50_to_pIC50(IC50):
    return -math.log10(IC50)

# ------------------------------
# Helper Functions
# ------------------------------
def filter_valid_smiles(smiles_list):
    valid, invalid = [], []
    for s in smiles_list:
        if Chem.MolFromSmiles(s):
            valid.append(s)
        else:
            invalid.append(s)
    return valid, invalid

def generate_descriptors_safe_individual(smiles_list):
    tmpdir = tempfile.mkdtemp()
    os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx4G"
    all_desc = []
    skipped = []

    progress_bar = st.progress(0)

    for i, smi in enumerate(smiles_list):
        try:
            batch_file = os.path.join(tmpdir, f"desc_{i}.csv")
            from_smiles([smi], output_csv=batch_file, fingerprints=True, timeout=600)
            if os.path.exists(batch_file):
                desc_df = pd.read_csv(batch_file)
                all_desc.append(desc_df)
            else:
                skipped.append(smi)
        except Exception:
            skipped.append(smi)
        progress_bar.progress((i+1)/len(smiles_list))

    shutil.rmtree(tmpdir, ignore_errors=True)

    if skipped:
        st.warning(f"{len(skipped)} SMILES could not be processed. Examples: {skipped[:5]}")

    return pd.concat(all_desc, ignore_index=True)

def run_prediction(model_key, smiles_input, uploaded):
    if smiles_input.strip():
        smiles_list = [line.strip() for line in smiles_input.strip().splitlines()]
    elif uploaded:
        smiles_list = [line.decode("utf-8").strip() for line in uploaded.readlines()]
    else:
        st.error("Please provide SMILES input or upload a file.")
        return

    smiles_list, invalid_smiles = filter_valid_smiles(smiles_list)
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES removed. Examples: {invalid_smiles[:5]}")
    if not smiles_list:
        st.error("No valid SMILES to process!")
        return

    with st.spinner("Generating descriptors using PaDELPy..."):
        desc_df = generate_descriptors_safe_individual(smiles_list)

    model = joblib.load(MODEL_PATHS[model_key])
    training_features = pd.read_csv(FEATURE_PATHS[model_key]).columns.tolist()

    common_features = [f for f in training_features if f in desc_df.columns]
    X = desc_df[common_features]
    if X.empty:
        st.error("No matching features found between descriptors and training data!")
        return

    with st.spinner(f"Predicting activity using {model_key}..."):
        prediction = model.predict(X)

    activity = [" Likely Inhibitor" if p >= 6 else " Likely Non-inhibitor" for p in prediction]
    ic50_values = [pIC50_to_IC50(p) for p in prediction]

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(
    """
    <h2 style='color:#006666; font-size:40px; font-family:"Times New Roman", serif; font-weight:bold;'>
        Prediction Results
    </h2>
    """,
    unsafe_allow_html=True
)

    # numeric results for CSV
    results_df = pd.DataFrame({
        "SMILES": desc_df["Name"],
        "pIC₅₀ value": prediction,
        "Prediction": activity
    })

    # ✅ display copy with pIC₅₀ converted to string → forces left alignment
    display_df = results_df.copy()
    display_df["pIC₅₀ value"] = display_df["pIC₅₀ value"].map(lambda x: f"{x:.4f}")

    # 🎨 Apply color styling based on Prediction column

    def style_table(row):
        base_color = '#f9ffff' if row.name % 2 == 0 else '#ffffff'
        styles = [f'background-color: {base_color}; color: #00332e; font-size:16px; text-align:center;'] * len(row)

        # Highlight Prediction column
        if row["Prediction"] == "Likely Inhibitor":
            styles[-1] = 'background-color: #006666; color: white; font-weight: bold; text-align:center;'
        elif row["Prediction"] == "Likely Non-inhibitor":
            styles[-1] = 'background-color: #cce6ff; color: #065f46; font-weight: bold; text-align:center;'

        return styles

    # ✅ Reset index so no 0,1,2,... column
    display_df = display_df.reset_index(drop=True)

    # ✅ First create styled DataFrame
    styled_df = display_df.style.apply(style_table, axis=1)

    # ✅ Then add header/cell styles
    styled_df = styled_df.set_table_styles(
        [
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#006666'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('font-size', '18px'),
                    ('padding', '12px')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '12px'),
                    ('text-align', 'center'),
                    ('border', '1px solid #ddd')
                ]
            }
        ]
    )

    # ✅ Inject global CSS
    st.markdown("""
    <style>
        table {
            width: 100% !important;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
        }
        table th {
            text-align: center !important;
        }
        table tr:hover {
            background-color: #e6ffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ✅ Render the styled table
    table_html = styled_df.to_html(index=False)
    # Wrap in scrollable container
    full_html = '<div style="width:100%; overflow-x:auto;">' + table_html + '</div>'

# Render
    st.markdown(full_html, unsafe_allow_html=True)

    # ✅ keep numeric values in CSV
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"{model_key}_predictions.csv",
        mime="text/csv"
    )


# ------------------------------
# Main Title
# ------------------------------
st.markdown("<div class='main-title'>MDM2InPred: Prediction of MDM2 Inhibitors</div>", unsafe_allow_html=True)
# ------------------------------
# Tabs
# ------------------------------
# Custom CSS for tabs font size
st.markdown("""
<style>
/* Increase tab font size */
.stTabs [role="tab"] p {
    font-size: 20px !important;   /* adjust size */
    font-family: times new roman;
    padding-bottom: 0px !important;
}
        
</style>
""", unsafe_allow_html=True)

# Tabs
tab_home, tab_pred, tab_con, tab_data, tab_help, tab_contact = st.tabs(
    ["Home", "Prediction", "Converter", "Dataset", "Help", "Contact"]
)

# ------------------------------
# HOME
# ------------------------------
with tab_home:
    col1, col2 = st.columns([1.3, 2])

    # Left side (Image)
    with col1:
        st.markdown(
            """
            <style>
            .custom-img img {
                height: 5000px;   /* make image taller */
                border-radius: 12px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
                margin-left: 200px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="custom-img">', unsafe_allow_html=True)
        st.image("img1.jpg", use_container_width=True, output_format="PNG")
        st.markdown('</div>', unsafe_allow_html=True)

    # Right side (Intro Paragraph)
    with col2:
        st.markdown(
            """
            <style>
            .intro-box {
                font-size: 18px;
                line-height: 1.6;
                text-align: justify;
                margin-top: 90px;
                margin-left: 100px;
                font-family:book antiqua;
            }

            </style>
            <div class="intro-box">
                Murine double minute 2 (MDM2) is a p53-specific E3 ubiquitin ligase that regulates 
                the cell cycle, DNA repair, apoptosis, and oncogene activation through both p53-dependent 
                and independent pathways. MDM2 has emerged as the primary cellular antagonist of p53. 
                Interestingly, MDM2 is itself a product of a p53-inducible gene, and the two are connected 
                through an autoregulatory negative feedback loop that keeps p53 levels low in unstressed cells.  
                MDM2 binds directly to the N-terminal transactivation domain of p53, blocking its transcriptional 
                activity, which leads to nuclear export of p53, followed by ubiquitination and directing it to 
                the 26S proteasome for subsequent proteasomal degradation. Under oncogenic stress, ARF sequesters 
                MDM2 in the nucleolus, preventing p53 degradation and enabling the transcription of genes such as 
                p21 (cell cycle arrest), BAX, and PUMA (apoptosis).  
                MDM2 overexpression is observed in multiple cancers, underscoring its carcinogenic potential 
                and therapeutic relevance. In cancer pharmacology, small-molecule inhibitors are designed to 
                prevent the complex formation between MDM2 and p53 by blocking MDM2’s binding site, thereby 
                restoring p53’s function.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Module 1: Prediction
    st.markdown("""
    <style>
    .module-box1 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: 25px;
        margin-left: 150px;
        font-family:book antiqua;
    }
                
    .module-box2 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: -200px;
        margin-left: 700px;
        font-family:book antiqua;
    }
                 
    .module-title {
        font-size: 24px;
        font-weight: bold;
        color: #006666;
        margin-bottom: 10px;
        text-align: center;
    }
    .module-text {
                font-size: 16px;
                color: #333;
                margin-bottom: 15px;
                text-align: justify;
            }
                
    .module {
        background: #ffffff;
        height: 60px; 
        width: 1270px;    
        margin-top: -10px;
    }
                
    .main-title {
                font-size: 40px;
                font-weight: bold;
                color: #006666;
                text-align: center;
    }
            </style>
            """,
            unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module">
            <div class="main-title">Modules</div>
        </div>
        """,unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-box1">
            <div class="module-title">Prediction</div>
            <div class="module-text">
                It enable users to predict the pIC50 value of query
                and whether it is an inhibitor or non-inhibitor of MDM2.
                The user can either paste SMILES strings directly or upload a .smi file upto 200MB in size
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Module 2: Converter
    st.markdown(
        """
        <div class="module-box2">
            <div class="module-title">Converter</div>
            <div class="module-text">
                It enables user bidirectional conversion between IC50(in M)
                and pIC50. Users can convert the predicted output obtained 
                in pIC50 to IC50 using this module.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# PREDICTION
# ------------------------------
with tab_pred:
    st.markdown("""
    <style>
    .prediction-box {
            background-color: #e6ffff;
            height: 180px;
            width: 1270px;
            padding: 0px;
            margin-bottom: 30px;
            margin-top: 15px;
            }
                
    .prediction-text {
            font-size: 18px;
            text-align: justify;
            font-family:book antiqua;
            }
                
    .prediction-title {
            font-size: 30px;
            text-align: center;
            color: #006666;
            font-weight: bold;
            font-family: zumila;
            }
                
    </style>
    """,unsafe_allow_html=True)
    
    st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-title">MDM2 Prediction Module</div>
                        <div class="prediction-text">
                            This modules predict the pIC50 value of query and also predicts whether it is
                            an inhibitor or non-inhibitor of MDM2. Here the user can use Random Forest (RF) 
                            or Light Gradinet Boosting Machine (LightGBM) machine learning algorithm has been 
                            implemented in the backend and user can select between both of the for prediction.
                            The result will be visible in tablular format and can also be downlaodable in CSV 
                            file format. For more information, Please refer to the Help page.
                        </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("---")

    st.markdown("""
    <style>
    /* Style for st.radio label (Select Model:) */
    div[data-testid="stRadio"] p {
        font-size: 16px !important;
        font-family: zumila;
        font-weight: normal !important;
        color: #000000 !important;
    }

    /* Style for st.write text like "You selected:" */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 16px !important;
        font-family: zumila;
        font-weight: normal !important;
        color: #000000 !important;
    }

    /* Style for st.text_area label (Paste SMILES string(s):) */
    div[data-testid="stTextArea"] label p {
        font-size: 16px !important;
        font-weight: normal !important;
        font-family: zumila;
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)


    # Choose model with radio button
    model_choice = st.radio(
        "Select Model:",
        ["LightGBM", "Random Forest"],
        horizontal=True
    )
    st.write("You selected:", model_choice)

    # Custom CSS for text area
    st.markdown("""
    <style>
    /* Style text area */
    [data-testid="stTextArea"] textarea {
        background-color: #f0ffff !important;  /* light cyan */
        color: #003333 !important;             /* dark teal text */
        font-size: 20px !important;
        font-family: "Courier New", monospace !important;
        border: 2px solid #006666 !important;
        border-radius: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # SMILES input
    smiles_input = st.text_area("Paste SMILES string(s):", key="smiles_input")


    # Example text changes depending on model
    if "LightGBM" in model_choice:
        st.markdown(
            '<p class="example-text">Example: CC1=CC=C(C=C1)N2C(=O)N=C(S2)NC3=CC=CC=C3</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="example-text">Example: COC1=CC=CC=C1O</p>',
            unsafe_allow_html=True
        )

    # Custom CSS for file uploader
    st.markdown("""
    <style>
    /* Style file uploader box */
    [data-testid="stFileUploader"] section {
        background-color: #e6ffff !important;  /* light cyan background */
        border: 2px dashed #006666 !important; /* dashed teal border */
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* Style text inside uploader */
    [data-testid="stFileUploader"] label {
        color: #003333 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded = st.file_uploader("Or upload a .smi file", type=["smi"], key="file_upload")


    # Prediction button
    if st.button("Run Prediction"):
        if "LightGBM" in model_choice:
            run_prediction("Model 1", smiles_input, uploaded)
        else:
            run_prediction("Model 2", smiles_input, uploaded)

# ------------------------------
# CONVERTER
# ------------------------------
with tab_con:
    st.markdown("""
    <style>
    .prediction-box {
            background: #ffffff;
            height: 140px;
            width: 1270px;
            padding: 0px;
            margin-bottom: 15px;
            margin-top: 15px;
            }
                
    .prediction-text {
            font-size: 16px;
            text-alight: justify;
            }
                
    .prediction-title {
            font-size: 30px;
            text-align: center;
            color: #006666;
            font-weight: bold;
            }
                
    /* Style text input box */
    [data-testid="stTextInput"] input {
        background-color: #f0ffff !important;  /* light cyan */
        color: #003333 !important;             /* dark teal text */
        font-size: 16px !important;
        font-family: "Courier New", monospace !important;
        border: 2px solid #006666 !important;
        border-radius: 0px !important;
    }
    
    </style>
    """,unsafe_allow_html=True)

    
    st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-title">Converter Module</div>
                        <div class="prediction-text">
                            This module was developed for bidirectional conversion between IC50(in M) and pIC50.
                            User can select the conversion type and can obtain the result. For more information,
                            Please refer to the Help page.
                        </div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")

    conversion_type = st.radio("Select conversion type:", ["pIC₅₀ ➝ IC₅₀", "IC₅₀ ➝ pIC₅₀"])

    if conversion_type == "pIC₅₀ ➝ IC₅₀":
        pic50_value = st.text_input("Enter pIC₅₀ value:")
        if st.button("Convert to IC₅₀"):
            try:
                val = float(pic50_value)
                ic50_value = pIC50_to_IC50(val)
                st.success(f"IC₅₀ value: {ic50_value:.6e} M")
            except:
                st.error("Invalid input.")

    else:
        ic50_value = st.text_input("Enter IC₅₀ value (in M):")
        if st.button("Convert to pIC₅₀"):
            try:
                val = float(ic50_value)
                if val > 0:
                    pic50_value = IC50_to_pIC50(val)
                    st.success(f"pIC₅₀ value: {pic50_value:.4f}")
                else:
                    st.error("IC₅₀ must be > 0")
            except:
                st.error("Invalid input.")


# ------------------------------
# DATASET
# ------------------------------
import streamlit as st
import pandas as pd
import os

with tab_data:
# Put this before you create the buttons (inside with tab_data:)
    st.markdown("""
    <style>
    /* Base button style */
    div.stButton > button {
        background-color: #e0ffff !important;   /* light cyan */
        color: #004d4d !important;              /* dark text */
        border: 2px solid #004d4d !important;
        border-radius: 8px;
        font-weight: bold;
        padding: 8px 20px;
        transition: all 0.2s ease-in-out;
    }

    /* Hover effect */
    div.stButton > button:hover {
        background-color: #66b2b2 !important;   /* medium teal */
        color: white !important;
    }

    /* Active/Clicked effect */
    div.stButton > button:active {
        background-color: #004d4d !important;   /* dark teal */
        color: #ffffff !important;
        border: 2px solid #002626 !important;
    }
    </style>
    """, unsafe_allow_html=True)


    st.markdown("""           
                    <div class="prediction-title">Dataset of MDM2InPred</div>
                    """, unsafe_allow_html=True)
    models = ["LightGBM", "Random Forest"]
    files = {
        "LightGBM": {
            "Training Set": "static/lightgbm_train_set.csv",
            "Test Set": "static/lightgbm_test_set.csv",
            "External Validation Set": "static/predicted_activity_external_lightgbm.csv"
        },
        "Random Forest": {
            "Training Set": "static/train_set.csv",
            "Test Set": "static/test_set.csv",
            "External Validation Set": "static/validation_set.csv"
        }
    }

    # ---- Table ----
    st.markdown(
        """
        <style>
        .dataset-table th {
            background-color: #006666;
            color: white;
            padding: 10px;
            text-align: center;
        }
        .dataset-table td {
            text-align: center;
            padding: 8px;
            border: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Make table layout with buttons
    for model in models:
        st.write(f"{model}")
        cols = st.columns(3)
        for i, dataset in enumerate(["Training Set", "Test Set", "External Validation Set"]):
            file_path = files[model][dataset]
            if cols[i].button(f"📂 {dataset}", key=f"{model}-{dataset}"):
                if os.path.exists(file_path):
                    st.subheader(f"📂 {model} - {dataset}")
                    df = pd.read_csv(file_path)
                    st.dataframe(df, use_container_width=True)
                    st.markdown(
                        """
                        <style>
                            [data-testid="stElementToolbar"] {
                            display: none;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.error(f"❌ File not found: {file_path}")

# ------------------------------
# HELP
# ------------------------------
with tab_help:
    st.markdown("""
    <style>
    .prediction-box {
            background: #ffffff;
            height: 50px;
            width: 1270px;
            padding: 0px;
            margin-bottom: 15px;
            margin-top: 0px;
            }
                
    .help-title {
            font-family: book antiqua;
            font-size: 20px;
            }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
                <div class="prediction-box">
                    <div class="prediction-text">
                            This dashboard is designed to provide an interactive interface for researchers to screen chemical 
                            compounds as potential MDM2 inhibitors and non-inhibitors virtually. It includes a prediction module 
                            for the prediction of MDM2 inhibitors and non-inhibitors, and an additional module for bidirectional 
                            conversion between IC50 and pIC50. The following video tutorial demonstrates how to navigate the 
                            dashboard and access its features.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    # ---- Add video ----
    st.video("video.mp4")   

    

# ------------------------------
# CONTACT
# ------------------------------
with tab_contact:

    import base64

    # ---------- Helper: Convert Image to Base64 ----------
    def get_base64_image(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # ---------- Styles ----------
    st.markdown("""
    <style>
    .section-title {
        font-size: 30px;
        text-align: center;
        color: #006666;
        font-weight: 700;
        margin: 20px 0 30px 0;
        font-family: 'Segoe UI', sans-serif;
    }
    .profile-card {
        background: #f9f9f9;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin: 10px;
    }
    .profile-card img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 10px;
        border: 3px solid #006666;
    }
    .profile-name {
        font-size: 16px;
        font-weight: 600;
        color: #006666;
        margin-bottom: 3px;
    }
    .profile-role {
        font-size: 14px;
        color: #333;
        margin-bottom: 3px;
    }
    .profile-email {
        font-size: 13px;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Contact Us Title ----------
    st.markdown("<div class='section-title'>Contact Us</div>", unsafe_allow_html=True)

    # ---------- Head Profile ----------
    head_img = get_base64_image("images/head.jpg")

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown(f"""
        <div class="profile-card">
            <img src="data:image/png;base64,{head_img}">
            <div class="profile-name">Dr. John Doe</div>
            <div class="profile-role">Head of Department</div>
            <div class="profile-email">johndoe@example.com</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Our Team Title ----------
    st.markdown("<div class='section-title'>Our Team</div>", unsafe_allow_html=True)

    # ---------- Team Members ----------
    team = [
      ("images/1.jpg","Riya Patel","Developer","riya@example.com"),
      ("images/2.jpg","Bob","Developer","bob@example.com"),
      ("images/3.jpg","Charlie","Developer","charlie@example.com"),
      ("images/4.jpg","Daisy","Developer","daisy@example.com"),
      ("images/5.jpg","Ethan","Developer","ethan@example.com"),
    ]

    for i in range(0, len(team), 3):   # 3 per row
        row = team[i:i+3]
        cols = st.columns(len(row))
        for col, (img, name, role, email) in zip(cols, row):
            with col:
                img_base64 = get_base64_image(img)
                st.markdown(f"""
                <div class="profile-card">
                    <img src="data:image/png;base64,{img_base64}">
                    <div class="profile-name">{name}</div>
                    <div class="profile-role">{role}</div>
                    <div class="profile-email">{email}</div>
                </div>
                """, unsafe_allow_html=True)
