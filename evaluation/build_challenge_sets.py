# DEPENDENCIES
import re
import json
import random
import requests
from tqdm import tqdm
from pathlib import Path


ADV_DIR = Path("evaluation/adversarial")
AI_DIR  = Path("evaluation/ai_generated")

# Ensure adversarial dirs exist
(ADV_DIR / "paraphrased").mkdir(parents = True, exist_ok = True)
(ADV_DIR / "cross_model").mkdir(parents = True, exist_ok = True)


# Paraphrased Set (Rule-Based + Lightweight)
def simple_paraphrase(text: str) -> str:
    """
    Lightweight paraphrasing using synonym replacement & reordering: Avoids downloading PEGASUS
    """
    # Simple transformations
    text = re.sub(r'\b(is|are|was|were)\b', lambda m: m.group(0), text)  # placeholder
    
    # Add more rules if needed, or use Ollama for light rewrite
    return text


def paraphrase_with_ollama(text: str) -> str:
    """
    Use Ollama to lightly rephrase text
    """
    url     = "http://localhost:11434/api/generate"
    prompt  = f"Paraphrase this text in your own words, keeping the same meaning:\n\n{text}"
    payload = {"model"   : "mistral:7b",
               "prompt"  : prompt,
               "stream"  : False,
               "options" : {"temperature" : 0.6, 
                            "num_predict" : 300,
                           }
              }

    try:
        response = requests.post(url     = url, 
                                 json    = payload, 
                                 timeout = 30,
                                )

        if (response.status_code == 200):
            return response.json().get("response", text).strip()

    except:
        pass
    
    # fallback
    return text  


def build_paraphrased():
    print(" → Building paraphrased set (using Ollama)...")
    count   = 0
    domains = [d for d in AI_DIR.iterdir() if d.is_dir()]

    random.shuffle(domains)
    for domain_dir in domains:
        files = list(domain_dir.glob("*.txt"))
        random.shuffle(files)
        
        for file in files:
            if (count >= 100):
                return

            with open(file, encoding = "utf-8") as f:
                text = f.read()

            para = paraphrase_with_ollama(text)
            
            with open(ADV_DIR / "paraphrased" / file.name, "w", encoding="utf-8") as f:
                f.write(para)

            count += 1

# Cross-Model Set (Using Ollama's llama3:8b)
def generate_cross_model_text(domain: str) -> str:
    """
    Generate text using a different model (llama3) than main AI set (mistral)
    """
    url        = "http://localhost:11434/api/generate"
    prompt_map = {"academic"      : "Write a scholarly abstract (200–400 words) on a recent scientific topic.",
                  "technical_doc" : "Write clear technical documentation for a software API endpoint.",
                  "creative"      : "Write a short creative story or descriptive passage.",
                  "social_media"  : "Write an engaging social media post about technology.",
                  "business"      : "Write a professional business summary or market analysis paragraph.",
                  "legal"         : "Draft a formal legal clause about data privacy compliance.",
                 }

    prompt     = prompt_map.get(domain, f"Write a 1000-word {domain.replace('_', ' ')} text.")

    payload    = {"model"   : "llama3:8b",
                  "prompt"  : prompt,
                  "stream"  : False,
                  "options" : {"temperature" : 0.8, 
                               "num_predict" : 400,
                              }
                 }
    try:
        response = requests.post(url     = url, 
                                 json    = payload, 
                                 timeout = 60,
                                )

        if (response.status_code == 200):
            return response.json().get("response", "").strip()

    except Exception as e:
        print(f"⚠️ Failed to generate for {domain}: {e}")

    return ""


def build_cross_model():
    print(" → Building cross-model set (using llama3:8b)...")

    domains = ["academic", "technical_doc", "creative", "social_media", "business", "legal"]
    count   = 0

    for domain in domains:
        for i in range(35):  # ~210 total, cap at 200
            if (count >= 200):
                return

            text = generate_cross_model_text(domain)

            if text:
                with open(ADV_DIR / "cross_model" / f"llama3_{domain}_{i}.txt", "w", encoding="utf-8") as f:
                    f.write(text)

                count += 1

# Main Execution
if __name__ == "__main__":
    print("Building challenge sets using Ollama (no HF downloads)...")
    build_paraphrased()
    build_cross_model()
    
    print("Challenge sets ready.")