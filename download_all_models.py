# filename: download_all_models.py
import time
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- List of Models to Download ---
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

all_model_names = list(FOUNDATION_MODELS.values()) + list(DOMAIN_MODELS.values())

def download_and_cache_models():
    print("🚀 Starting the download and caching process for all models...")
    start_time = time.time()

    for i, model_name in enumerate(all_model_names):
        print("-" * 60)
        print(f"Processing model {i+1}/{len(all_model_names)}: {model_name}")

        try:
            print(f"  -> Downloading tokenizer for {model_name}...")
            AutoTokenizer.from_pretrained(model_name)
            print(f"  ✅ Tokenizer for {model_name} cached successfully.")

            print(f"  -> Downloading model weights for {model_name}...")
            AutoModelForMaskedLM.from_pretrained(model_name)
            print(f"  ✅ Model '{model_name}' cached successfully.")

        except Exception as e:
            print(f"  ❌ FAILED to download or cache {model_name}. Error: {e}")

    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 60)
    print(f"🎉 Finished processing all models in {total_time:.2f} seconds.")
    print("The models are now cached and ready for use in your Streamlit app.")

if __name__ == "__main__":
    download_and_cache_models()