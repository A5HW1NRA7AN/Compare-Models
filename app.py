# filename: app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# --- UI Configuration ---
st.set_page_config(
    page_title="NLP Model Comparison",
    layout="wide"
)

st.title("🎯 Foundation vs. Domain-Specific Model Comparison")
st.write("""
This app compares a general-purpose **foundation model** with a **domain-specific model** on a **Fill-Mask** task. 
Enter a sentence with a `[MASK]` token to see which model provides more contextually relevant predictions. The results are shown in both graphs and tables.
""")

# --- Model Selection ---
FOUNDATION_MODELS = {
    "BERT (Base, Uncased)": "bert-base-uncased",
    "RoBERTa (Base)": "roberta-base",
    "DistilBERT (Base, Uncased)": "distilbert-base-uncased",
}

DOMAIN_MODELS = {
    "BioClinicalBERT (Medical)": "emilyalsentzer/Bio_ClinicalBERT",
    "SciBERT (Scientific)": "allenai/scibert_scivocab_uncased",
    "FinBERT (Financial)": "ProsusAI/finbert",
    "LegalBERT (Legal)": "nlpaueb/legal-bert-base-uncased",
}

SAMPLE_TEXTS = {
    "Medical": "A patient with chronic kidney disease requires weekly [MASK].",
    "Scientific": "New research in [MASK] shows promising results for cancer treatment.",
    "Financial": "The company's stock price saw a significant [MASK] after the earnings report.",
    "Legal": "The defendant was found [MASK] by the jury.",
    "General": "The best way to learn a new language is to practice it [MASK]."
}

# --- Caching ---
@st.cache_resource(show_spinner="Loading models...")
def load_fill_mask_pipeline(model_name):
    """Loads a fill-mask pipeline for a given model name."""
    return pipeline("fill-mask", model=model_name)

# --- Sidebar Layout ---
st.sidebar.header("⚙️ Model Selection")
foundation_choice_key = st.sidebar.selectbox(
    "Choose a Foundation Model:",
    options=list(FOUNDATION_MODELS.keys())
)
foundation_model_name = FOUNDATION_MODELS[foundation_choice_key]

domain_choice_key = st.sidebar.selectbox(
    "Choose a Domain-Specific Model:",
    options=list(DOMAIN_MODELS.keys())
)
domain_model_name = DOMAIN_MODELS[domain_choice_key]

# --- Input Section ---
st.header("✍️ Input Text")
input_option = st.radio(
    "Choose an input method:",
    ["Select from samples", "Enter manually"],
    horizontal=True,
    label_visibility="collapsed"
)

if input_option == "Select from samples":
    sample_category = st.selectbox("Select a sample sentence domain:", list(SAMPLE_TEXTS.keys()))
    input_text = st.text_area(
        "Input sentence with `[MASK]` token:",
        value=SAMPLE_TEXTS[sample_category],
        height=100
    )
else:
    input_text = st.text_area(
        "Input sentence with `[MASK]` token:",
        value="The patient's symptoms suggest a diagnosis of [MASK].",
        height=100
    )

# --- Run & Display Results ---
if st.button("🚀 Compare Models"):
    if not input_text or "[MASK]" not in input_text:
        st.error("Please provide an input sentence containing the `[MASK]` token.")
    else:
        pipe_foundation = load_fill_mask_pipeline(foundation_model_name)
        pipe_domain = load_fill_mask_pipeline(domain_model_name)

        with st.spinner("Generating predictions..."):
            predictions_foundation = pipe_foundation(input_text)
            predictions_domain = pipe_domain(input_text)

        st.header("📊 Comparison of Top Predictions")
        st.info(f"**Your Input:** {input_text.replace('[MASK]', '**[MASK]**')}")

        col1, col2 = st.columns(2)

        def display_results(pipe_result, title, color_scale):
            df = pd.DataFrame(pipe_result)
            df_display = df.rename(columns={"score": "Confidence", "token_str": "Predicted Token", "sequence": "Filled Sentence"})
            df_display['Confidence'] = df_display['Confidence'].map('{:.2%}'.format)

            fig = px.bar(df, x="token_str", y="score", title=title,
                         labels={"token_str": "Predicted Token", "score": "Confidence Score"},
                         color="score", color_continuous_scale=color_scale, text_auto='.2%')
            fig.update_layout(xaxis_title="", yaxis_title="Confidence")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_display[['Predicted Token', 'Confidence', 'Filled Sentence']], use_container_width=True, hide_index=True)

        with col1:
            st.subheader(f"Results from `{foundation_choice_key}`")
            display_results(predictions_foundation, f"Top Predictions by {foundation_choice_key}", px.colors.sequential.Blues_r)

        with col2:
            st.subheader(f"Results from `{domain_choice_key}`")
            display_results(predictions_domain, f"Top Predictions by {domain_choice_key}", px.colors.sequential.Greens_r)