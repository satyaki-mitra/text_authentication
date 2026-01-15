# DEPENDENCIES
import json
import requests
from tqdm import tqdm
from pathlib import Path


# Use Ollama models you already have
MODEL_NAME = "mistral:7b"  # or "llama3:8b"

PROMPTS    = {"general"       : "Write a general informative paragraph on a common topic.",
              "academic"      : "Write a scholarly abstract in academic style (1000–2000 words).",
              "creative"      : "Write a short creative narrative or descriptive passage.",
              "ai_ml"         : "Write a technical abstract about a machine learning concept.",
              "software_dev"  : "Write a developer-focused explanation of a programming tool or pattern.",
              "technical_doc" : "Write clear technical documentation for a software feature.",
              "engineering"   : "Write a concise engineering report excerpt on a system design.",
              "science"       : "Write a scientific explanation of a natural phenomenon.",
              "business"      : "Write a professional business summary or market analysis.",
              "legal"         : "Write a formal legal clause or case summary in precise language.",
              "medical"       : "Write a clinical abstract or medical case description.",
              "journalism"    : "Write a journalistic news article excerpt in neutral tone.",
              "marketing"     : "Write persuasive marketing copy for a tech product.",
              "social_media"  : "Write an informal social media post or comment.",
              "blog_personal" : "Write a personal blog post sharing an opinion or experience.",
              "tutorial"      : "Write a step-by-step tutorial for beginners.",
             }

HUMAN_DIR  = Path("evaluation/human")
AI_DIR     = Path("evaluation/ai_generated")

def generate_with_ollama(prompt: str, max_tokens=512) -> str:
    url     = "http://localhost:11434/api/generate"
    payload = {"model"   : MODEL_NAME,
               "prompt"  : prompt,
               "stream"  : False,
               "options" : {"temperature" : 0.7,
                            "top_p"       : 0.9,
                            "num_predict" : max_tokens,
                           }
              }

    try:
        response = requests.post(url     = url, 
                                 json    = payload, 
                                 timeout = 60,
                                )

        if (response.status_code == 200):
            result = response.json()
            return result.get("response", "").strip()

        else:
            print(f"⚠️ Ollama error: {response.status_code}")
            return ""

    except Exception as e:
        print(f"⚠️ Request failed: {e}")
        return ""


def main():
    for domain_dir in HUMAN_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name
        
        if domain not in PROMPTS:
            continue

        (AI_DIR / domain).mkdir(parents = True, exist_ok = True)

        files = list(domain_dir.glob("*.txt"))

        print(f"Generating AI texts for {domain} ({len(files)} samples)...")

        for i, file in enumerate(tqdm(files, desc = f"→ {domain}")):
            prompt  = PROMPTS[domain]
            ai_text = generate_with_ollama(prompt)
            
            if ai_text:
                with open(AI_DIR / domain / f"ai_{domain}_{i}.txt", "w", encoding = "utf-8") as f:
                    f.write(ai_text)
            
            else:
                # Fallback: duplicate human text (won't affect eval much if rare)
                with open(file, "r") as f:
                    fallback = f.read()
                
                with open(AI_DIR / domain / f"ai_{domain}_{i}.txt", "w") as f:
                    # Truncate to avoid leakage
                    f.write(fallback[:2000]) 


# Execution
if __name__ == "__main__":
    print(f"Generating AI texts using Ollama ({MODEL_NAME})")
    main()
    print("AI-generated data created for all domains.")