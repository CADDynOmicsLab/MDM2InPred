import streamlit as st
import pandas as pd
import joblib
import tempfile
import shutil
import os
import math
import base64
import cohere
from huggingface_hub import hf_hub_download

# ------------------------------
# Import required libraries
# ------------------------------
try:
    from rdkit import Chem
    from padelpy import from_smiles
except ImportError:
    st.error("Required packages not found. Please install rdkit and padelpy.")
    st.stop()

# ------------------------------
# Page Config (MUST BE FIRST)
# ------------------------------
st.set_page_config(page_title="MDM2 pIC50 Prediction", layout="wide")

# ------------------------------
# SAFE Session State Initialization
# ------------------------------
@st.cache_resource(experimental_allow_widgets=True)
def init_session_state():
    return {
        "smiles_input": "",
        "chat_history": []
    }

# Initialize if not exists
for key, value in init_session_state().items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------------------------
# Initialize session state (with error handling)
# ------------------------------
try:
    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = ""
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
except Exception as e:
    # Initialize with default values if session state isn't ready
    st.session_state.smiles_input = ""
    st.session_state.chat_history = []

# ------------------------------
# CONFIG - Update paths
# ------------------------------
FEATURE_PATHS = {
    "Model 1": "newm1_aligned_376.csv",
    "Model 2": "rf_ready_newm2.csv"
}
DATASET_PATH = "main.csv"

# ------------------------------
# Google Fonts
# ------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ------------------------------
# Custom CSS (Theme) - FIXED with !important
# ------------------------------
st.markdown("""
<style>
/* Global font & background */
.stApp {
    background: #ffffff !important;
    color: #1a1a1a !important;
}

/* Title */
.main-title {
    font-size: 42px !important;
    font-weight: 700 !important;
    color: #006666 !important;
    text-align: center !important;
    font-family: 'Playfair Display', serif !important;
    margin-bottom: 10px !important;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.15) !important;
    letter-spacing: 0.5px !important;
}

/* Tabs container */
.stTabs [role="tablist"] {
    display: flex !important;
    justify-content: center !important;
    gap: 6px !important;
    border-bottom: 2px solid #006666 !important;
    flex-wrap: nowrap !important;
}

/* Individual tab */
.stTabs [role="tab"] {
    padding: 30px 28px !important;
    background-color: #e6ffff !important;
    border-radius: 10px 10px 0 0 !important;
}

/* Tab text */
.stTabs [role="tab"] span {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #006666 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background-color: #006666 !important;
    color: white !important;
    box-shadow: 0px -3px 8px rgba(0,0,0,0.2) !important;
    transform: none !important;
}

/* Active tab text color */
.stTabs [aria-selected="true"] span {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background-color: #006666 !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    padding: 6px 20px !important;
}

.stButton>button:hover {
    background-color: #002244 !important;
}

/* Small example text */
.example-text {
    font-size: 20px !important;
    color: #444 !important;
    font-style: italic !important;
}

/* Converter box */
.converter-box {
    background-color: #e6ffff !important;
    padding: 20px !important;
    margin-bottom: 30px !important;
    border-radius: 10px !important;
}

.converter-title {
    font-size: 30px !important;
    text-align: center !important;
    color: #006666 !important;
    font-weight: bold !important;
    margin-bottom: 10px !important;
}

.converter-text {
    font-size: 16px !important;
    line-height: 1.6 !important;
    text-align: left !important;
}

/* Chat title */
.chat-title {
    font-size: 32px !important;
    color: #006666 !important;
    text-align: center !important;
    font-weight: bold !important;
    margin-bottom: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Cohere Client (Render-safe)
# ------------------------------
co = cohere.Client(st.secrets["COHERE_API_KEY"])

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
@st.cache_resource
def load_model_from_hf(filename):
    try:
        model_path = hf_hub_download(
            repo_id="riya-patel/MDM2InPred-models",
            filename=filename,
            token=os.getenv("HF_TOKEN", "")
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model {filename}: {str(e)}")
        return None

def generate_descriptors_safe_individual(smiles_list):
    if not smiles_list:
        return pd.DataFrame()
        
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
        progress_bar.progress((i + 1) / len(smiles_list))

    shutil.rmtree(tmpdir, ignore_errors=True)

    if skipped:
        st.warning(f"{len(skipped)} SMILES could not be processed. Examples: {skipped[:5]}")

    if all_desc:
        return pd.concat(all_desc, ignore_index=True)
    else:
        return pd.DataFrame()

def run_prediction(model_key, smiles_input_text, uploaded_file):
    """
    Main prediction function
    """
    # ------------------------------
    # Read input (textarea OR file)
    # ------------------------------
    smiles_list = []
    
    if smiles_input_text and smiles_input_text.strip():
        smiles_list = [line.strip() for line in smiles_input_text.strip().splitlines()]
    elif uploaded_file is not None:
        try:
            raw_lines = [
                line.decode("utf-8", errors="ignore").strip()
                for line in uploaded_file.readlines()
            ]
            smiles_list = [
                l for l in raw_lines
                if l and not l.startswith("#")
            ]
        except Exception as e:
            st.error(f"Error reading uploaded file: {str(e)}")
            return
    else:
        st.error("Please paste SMILES or upload a valid .smi file.")
        return

    # ------------------------------
    # Clean and validate SMILES
    # ------------------------------
    smiles_list = [s.strip() for s in smiles_list if s.strip()]

    if not smiles_list:
        st.error("No SMILES strings found.")
        return

    cleaned = []
    rejected = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            cleaned.append(s)
        else:
            rejected.append(s)

    if rejected:
        st.warning(
            f"{len(rejected)} invalid SMILES removed. "
            f"Examples: {rejected[:5]}"
        )

    if not cleaned:
        st.error("No valid SMILES found. Please enter chemical SMILES only.")
        return

    smiles_list = cleaned

    # Limit for Render / Streamlit
    MAX_SMILES = 30
    if len(smiles_list) > MAX_SMILES:
        st.error("Maximum 30 SMILES allowed.")
        return

    # ------------------------------
    # Descriptor generation
    # ------------------------------
    with st.spinner("Generating descriptors using PaDELPy..."):
        desc_df = generate_descriptors_safe_individual(smiles_list)

    if desc_df.empty:
        st.error("Failed to generate descriptors. Please check your SMILES strings.")
        return

    # ------------------------------
    # Load model from Hugging Face
    # ------------------------------
    if model_key == "Model 1":
        model = load_model_from_hf("lightgbm.pkl")
    else:
        model = load_model_from_hf("rf.pkl")

    if model is None:
        st.error("Failed to load model. Please try again.")
        return

    # Load feature columns
    try:
        training_features = pd.read_csv(FEATURE_PATHS[model_key]).columns.tolist()
    except Exception as e:
        st.error(f"Error loading feature file: {str(e)}")
        return

    common_features = [f for f in training_features if f in desc_df.columns]
    
    if not common_features:
        st.error("No matching features found between descriptors and training data!")
        return
    
    X = desc_df[common_features]

    with st.spinner(f"Predicting activity using {model_key}..."):
        try:
            prediction = model.predict(X)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

    activity = ["Likely Inhibitor" if p >= 6 else "Likely Non-inhibitor" for p in prediction]

    # Results DataFrame
    results_df = pd.DataFrame({
        "SMILES": smiles_list,
        "pIC₅₀ value": prediction,
        "Prediction": activity
    })

    # Display copy with formatted pIC₅₀
    display_df = results_df.copy()
    display_df["pIC₅₀ value"] = display_df["pIC₅₀ value"].map(lambda x: f"{x:.4f}")

    # Apply color styling
    def style_table(row):
        base_color = '#f9ffff' if row.name % 2 == 0 else '#ffffff'
        styles = [f'background-color: {base_color}; color: #00332e; font-size:16px; text-align:center;'] * 3

        if row["Prediction"] == "Likely Inhibitor":
            styles[2] = 'background-color: #006666; color: white; font-weight: bold; text-align:center;'
        elif row["Prediction"] == "Likely Non-inhibitor":
            styles[2] = 'background-color: #cce6ff; color: #065f46; font-weight: bold; text-align:center;'

        return styles

    display_df = display_df.reset_index(drop=True)
    styled_df = display_df.style.apply(style_table, axis=1)

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

    # Display table
    st.markdown("""
    <style>
        table {
            width: 100% !important;
            border-collapse: collapse !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        table th {
            text-align: center !important;
        }
        table tr:hover {
            background-color: #e6ffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <h2 style='color:#006666; font-size:40px; font-family: Inter, sans-serif; font-weight:bold;'>
            Prediction Results
        </h2>
        """,
        unsafe_allow_html=True
    )

    table_html = styled_df.to_html(index=False)
    st.markdown('<div style="width:100%; overflow-x:auto;">' + table_html + '</div>', unsafe_allow_html=True)

    # Download button
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prediction Results (CSV)",
        data=csv_data,
        file_name=f"{model_key.replace(' ', '_')}_predictions.csv",
        mime="text/csv",
        key=f"download_{model_key}_{len(smiles_list)}"
    )

# ------------------------------
# Chatbot Context & Logic
# ------------------------------
APP_CONTEXT = """
This is the MDM2InPred dashboard.

Modules:

1) Home:
   - Describes the biological background of MDM2, its interaction with p53, and its role in cancer.
   - Explains the importance of small-molecule MDM2 inhibitors.

2) Prediction:
   - Predicts the pIC50 value of user-provided molecules and classifies them as MDM2 inhibitors or non-inhibitors.
   - User can either paste SMILES strings or upload a .smi file (up to 200 MB).
   - Two machine learning models are available: LightGBM and Random Forest.
   - Models are trained on molecular descriptors generated using PaDEL.

3) Converter:
   - Performs bidirectional conversion between IC50 (in M) and pIC50 using:
     pIC50 = -log10(IC50).
   - Users can input either pIC50 or IC50 and get the converted value.

4) Dataset:
   - Provides access to the training, test, and external validation sets used to develop the models.
   - Separate datasets are available for the LightGBM and Random Forest models.

5) Help:
   - Provides basic instructions on how to use the dashboard and includes a video tutorial.

6) Contact:
   - Shows contact information and team profiles for the developers or researchers.
"""

SYSTEM_INSTRUCTIONS = """
You are an assistant for the MDM2InPred Streamlit dashboard.
Use the APP CONTEXT to answer questions about:
- How to use each module (Prediction, Converter, Dataset, Help, Contact)
- General concepts: MDM2, p53, IC50, pIC50, inhibitors, SMILES, machine learning models (LightGBM, Random Forest).

RULES:
- If the user asks for exact internal data (CSV contents, weights, hidden parameters, or precise training details not in the context), say that you do not have direct access and ask them to check the Dataset or Prediction tab.
- Do NOT invent experimental results, exact pIC50 values, or dataset entries.
- If you are unsure, clearly say you are not sure and guide the user to the appropriate tab in the dashboard.
- Be clear, concise, and user-friendly.
"""

def chatbot_reply(user_text: str) -> str:
    prompt = f"""
System:
{SYSTEM_INSTRUCTIONS}

APP CONTEXT:
{APP_CONTEXT}

User question:
{user_text}
"""
    try:
        response = co.chat(
            model="command-a-03-2025",
            message=prompt,
            temperature=0.2,
            max_tokens=300
        )
        return response.text
    except Exception as e:
        return "Sorry, I could not generate an answer right now. Please try again later."

# ------------------------------
# Image Helper Function - FIXED
# ------------------------------
def load_image_safe(image_name):
    """Safely load image from multiple possible paths"""
    possible_paths = [
        image_name,
        f"./{image_name}",
        f"images/{image_name}",
        f"./images/{image_name}"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_base64_image(image_path):
    """Convert image to base64 - FIXED"""
    img_path = load_image_safe(image_path)
    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as f:
                data = f.read()
            return base64.b64encode(data).decode()
        except Exception as e:
            st.error(f"Error loading image {image_path}: {str(e)}")
    return ""

# ------------------------------
# Main Title
# ------------------------------
st.markdown("<div class='main-title'>MDM2InPred: Prediction of MDM2 Inhibitors</div>", unsafe_allow_html=True)

# ------------------------------
# Tabs
# ------------------------------
tab_home, tab_pred, tab_con, tab_data, tab_help, tab_contact, tab_chat = st.tabs(
    ["Home", "Prediction", "Converter", "Dataset", "Help", "Contact", "Ask AI"]
)

# ------------------------------
# HOME TAB - FIXED
# ------------------------------
with tab_home:
    col1, col2 = st.columns([1.3, 2])
    with col1:
        # Load image safely
        img_path = load_image_safe("img1.jpg")
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image 'img1.jpg' not found. Please add it to your project directory.")

    with col2:
        st.markdown(
            """
            <div style="font-size:18px; line-height:1.6; text-align:justify; margin-top:90px; margin-left:100px;">
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
                prevent the complex formation between MDM2 and p53 by blocking MDM2's binding site, thereby 
                restoring p53's function.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Module section
    st.markdown("""
    <style>
    .module-box1 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: 25px;
        margin-left: 150px;
    }
    .module-box2 {
        background: #ffffff;
        height: 200px; 
        width: 400px;
        padding: 20px;
        margin-top: -200px;
        margin-left: 700px;
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
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="module">
            <div class="main-title">Modules</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-box1">
            <div class="module-title">Prediction</div>
            <div class="module-text">
                It enables users to predict the pIC50 value of query
                and whether it is an inhibitor or non-inhibitor of MDM2.
                The user can either paste SMILES strings directly or upload a .smi file upto 200MB in size.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="module-box2">
            <div class="module-title">Converter</div>
            <div class="module-text">
                It enables bidirectional conversion between IC50 (in M)
                and pIC50. Users can convert the predicted output obtained 
                in pIC50 to IC50 using this module.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------
# PREDICTION TAB
# ------------------------------
with tab_pred:
    st.markdown("""     
        <div class="converter-box">
            <div class="converter-title">MDM2 Prediction Module</div>
            <div class="converter-text">
                This module predicts the pIC50 value of query molecules and also predicts whether they are
                inhibitors or non-inhibitors of MDM2. Light Gradient Boosting Machine (LightGBM) and 
                Random Forest (RF) machine learning algorithms have been implemented in the backend, and 
                users can select between both for prediction. The result will be visible in tabular format 
                and can also be downloaded as a CSV file. For more information, please refer to the Help page.
            </div>
        </div>
        """, unsafe_allow_html=True)

    model_choice = st.radio(
        "Select Model:",
        ["LightGBM", "Random Forest"],
        horizontal=True,
        key="model_choice_radio"
    )

    current_smiles = st.session_state.smiles_input
    
    smiles_input = st.text_area(
        "Paste SMILES string(s):",
        value=current_smiles,
        height=120,
        key="smiles_text_area",
        on_change=lambda: st.session_state.update({"smiles_input": st.session_state.smiles_text_area})
    )

    if smiles_input != st.session_state.smiles_input:
        st.session_state.smiles_input = smiles_input

    st.info("⚠️ Enter ONLY chemical SMILES (one per line). Do not paste code, text, or headings.")

    st.markdown(
        '<p class="example-text">Example: CC1=CC=C(C=C1)N2C(=O)N=C(S2)NC3=CC=CC=C3</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Or upload a .smi file", 
        type=["smi"], 
        key="file_upload"
    )
    
    if st.button("Run Prediction", key="run_prediction_btn"):
        current_input = st.session_state.get("smiles_text_area", "")
        
        if not current_input.strip() and uploaded is None:
            st.error("Please enter SMILES or upload a file.")
        else:
            if model_choice == "LightGBM":
                run_prediction("Model 1", current_input, uploaded)
            else:
                run_prediction("Model 2", current_input, uploaded)

# ------------------------------
# CONVERTER TAB
# ------------------------------
with tab_con:
    st.markdown("""
        <div class="converter-box">
            <div class="converter-title">Converter Module</div>
            <div class="converter-text">
                This module was developed for bidirectional conversion between IC50 (in M) and pIC50.
                Users can select the conversion type and obtain the result. For more information,
                please refer to the Help page.
            </div>
        </div>
        """, unsafe_allow_html=True)

    conversion_type = st.radio(
        "Select conversion type:", 
        ["pIC₅₀ ➝ IC₅₀", "IC₅₀ ➝ pIC₅₀"],
        key="converter_radio"
    )

    if conversion_type == "pIC₅₀ ➝ IC₅₀":
        pic50_value = st.text_input("Enter pIC₅₀ value:", key="pic50_input")
        if st.button("Convert to IC₅₀", key="convert_pic50"):
            try:
                val = float(pic50_value)
                ic50_value = pIC50_to_IC50(val)
                st.success(f"IC₅₀ value: {ic50_value:.6e} M")
            except:
                st.error("Invalid input.")
    else:
        ic50_value = st.text_input("Enter IC₅₀ value (in M):", key="ic50_input")
        if st.button("Convert to pIC₅₀", key="convert_ic50"):
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
# DATASET TAB
# ------------------------------
with tab_data:
    st.markdown("""           
        <div style="font-size:30px; text-align:center; color:#006666; font-weight:bold;">
            Dataset of MDM2InPred
        </div>
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

    for model in models:
        st.write(f"**{model}**")
        cols = st.columns(3)
        for i, dataset in enumerate(["Training Set", "Test Set", "External Validation Set"]):
            file_path = files[model][dataset]
            button_key = f"{model}_{dataset.replace(' ', '_')}_btn_{i}"
            if cols[i].button(f"📂 {dataset}", key=button_key):
                if os.path.exists(file_path):
                    st.subheader(f"📂 {model} - {dataset}")
                    df = pd.read_csv(file_path)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"❌ File not found: {file_path}")

# ------------------------------
# HELP TAB
# ------------------------------
with tab_help:
    st.markdown("""
        <div style="font-size:16px; line-height:1.6; margin-bottom:20px;">
            This dashboard is designed to provide an interactive interface for researchers to screen chemical 
            compounds as potential MDM2 inhibitors and non-inhibitors virtually. It includes a prediction module 
            for the prediction of MDM2 inhibitors and non-inhibitors, and an additional module for bidirectional 
            conversion between IC50 and pIC50. The following video tutorial demonstrates how to navigate the 
            dashboard and access its features.
        </div>
        """, unsafe_allow_html=True)

    video_path = load_image_safe("video.mp4")
    if video_path:
        st.video(video_path)
    else:
        st.warning("Video file 'video.mp4' not found.")

# ------------------------------
# CONTACT TAB - FIXED
# ------------------------------
with tab_contact:
    st.markdown("""
    <style>
    .section-title {
        font-size: 30px !important;
        text-align: center !important;
        color: #006666 !important;
        font-weight: 700 !important;
        margin: 20px 0 30px 0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .profile-card {
        background: #f9f9f9 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        text-align: center !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08) !important;
        margin: 10px !important;
    }
    .profile-card img {
        width: 120px !important;
        height: 120px !important;
        border-radius: 50% !important;
        object-fit: cover !important;
        margin-bottom: 10px !important;
        border: 3px solid #006666 !important;
    }
    .profile-name {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #006666 !important;
        margin-bottom: 3px !important;
    }
    .profile-role {
        font-size: 14px !important;
        color: #333 !important;
        margin-bottom: 3px !important;
    }
    .profile-email {
        font-size: 13px !important;
        color: #555 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Contact Us</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-name">Dr. Sarfaraz Alam</div>
            <div class="profile-role">Associate Professor</div>
            <div class="profile-role">CADDynOmics Lab</div>
            <div class="profile-role">Institute of Advanced Research, The University for Innovation, Gandhinagar</div>
            <div class="profile-email">sarfaraz.alam@iar.ac.in</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Our Team</div>", unsafe_allow_html=True)

    team = [
        ("images/1.jpg", "Zarnalipi soren", "P.HD", "riya@example.com"),
        ("images/riyaa.jpg", "Riya Patel", "M.Sc data science", "riya20.surat@gmail.com"),
        ("images/pranjal.jpeg", "Pranjal Oza", "M.Sc data science", "pranjaloza7@gmail.com"),
        ("images/meet.jpeg", "Meet Bhayani", "M.Sc data science", "meetmbhayani@gmail.com"),
        ("images/raish.jpeg", "Raishbhai Mansuri", "M.Sc data science", "raishmansuri2003@gmail.com"),
    ]

    for i in range(0, len(team), 3):
        row = team[i:i+3]
        cols = st.columns(len(row))
        for col, (img, name, role, email) in zip(cols, row):
            with col:
                img_base64 = get_base64_image(img)
                if img_base64:
                    st.markdown(f"""
                    <div class="profile-card">
                        <img src="data:image/png;base64,{img_base64}">
                        <div class="profile-name">{name}</div>
                        <div class="profile-role">{role}</div>
                        <div class="profile-email">{email}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="profile-card">
                        <div style="width:120px;height:120px;background:#ddd;border-radius:50%;margin:0 auto 10px;"></div>
                        <div class="profile-name">{name}</div>
                        <div class="profile-role">{role}</div>
                        <div class="profile-email">{email}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ------------------------------
# CHATBOT TAB
# ------------------------------
with tab_chat:
    st.markdown("<div class='chat-title'>MDM2InPred Assistant</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style="font-size:16px; line-height:1.6; text-align:justify;">
        This assistant is designed to help you understand and use the MDM2InPred dashboard.
        You can ask questions about:
        </p>
        <ul style="font-size:16px; line-height:1.6;">
            <li>How to use the <b>Prediction</b> module (SMILES input, model selection, CSV output).</li>
            <li>How the <b>Converter</b> works for IC₅₀ and pIC₅₀ values.</li>
            <li>What information is available in the <b>Dataset</b> tab.</li>
            <li>General concepts such as MDM2, p53, inhibitors, IC₅₀, pIC₅₀, LightGBM and Random Forest.</li>
        </ul>
        <p style="font-size:16px; line-height:1.6; text-align:justify;">
        Type your question below and the assistant will respond using the information and context
        of this dashboard.
        </p>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### Conversation")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**Assistant:** {msg['text']}")

    st.markdown("---")

    user_input = st.text_input("Type your question here:", key="chat_input_field")

    col_send, col_clear = st.columns([1, 1])
    with col_send:
        send_btn = st.button("Send", key="send_chat_button")
    with col_clear:
        clear_btn = st.button("Clear Chat", key="clear_chat_button")

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    if send_btn and user_input.strip():
        text = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "text": text})
        bot_answer = chatbot_reply(text)
        st.session_state.chat_history.append({"role": "bot", "text": bot_answer})
        st.rerun()