# DEPENDENCIES
import json
import requests
from tqdm import tqdm
from pathlib import Path
from loguru import logger



# Configuration
MODEL_NAME = "mistral:7b"
HUMAN_DIR  = Path("evaluation/human")
AI_DIR     = Path("evaluation/ai_generated")

# Domain-specific generation parameters
GENERATION_PARAMS = {"academic"      : {"temperature" : 0.5, "top_p" : 0.85},
                     "creative"      : {"temperature" : 0.9, "top_p" : 0.95},
                     "legal"         : {"temperature" : 0.3, "top_p" : 0.80},
                     "social_media"  : {"temperature" : 0.8, "top_p" : 0.95},
                     "blog_personal" : {"temperature" : 0.8, "top_p" : 0.95},
                     "marketing"     : {"temperature" : 0.7, "top_p" : 0.90},
                     "journalism"    : {"temperature" : 0.6, "top_p" : 0.85},
                     "default"       : {"temperature" : 0.7, "top_p" : 0.9},
                    }

# Improved domain-specific prompts
PROMPTS           = {"general"       : "Write a comprehensive 500-1000 word informative article explaining a common topic that would appear in an encyclopedia. Use clear, neutral language with proper structure.",
                     "academic"      : "Write a formal research abstract (500-1000 words) for a scientific study. Include: background context, research methodology, key findings, and implications. Use academic vocabulary and formal tone appropriate for peer-reviewed publication.", 
                     "creative"      : "Write a creative narrative passage (500-1000 words) with vivid descriptions, engaging storytelling, and literary devices. Focus on character, setting, or emotion with rich sensory details.", 
                     "ai_ml"         : "Write a technical explanation (500-1000 words) of a machine learning concept or recent AI advancement. Include mathematical intuition, practical applications, and current research directions. Use precise technical terminology.",
                     "software_dev"  : "Write developer documentation (500-1000 words) explaining the implementation of a software design pattern or architectural principle. Include code examples, use cases, trade-offs, and best practices for professional developers.", 
                     "technical_doc" : "Write comprehensive API documentation (500-1000 words) for a REST endpoint. Include: endpoint description, request/response parameters with data types, authentication requirements, example requests, error codes, and usage notes. Use Markdown formatting.", 
                     "engineering"   : "Write an engineering technical report excerpt (500-1000 words) analyzing a system design or technical solution. Include specifications, performance analysis, design constraints, and recommendations.", 
                     "science"       : "Write a scientific explanation (500-1000 words) of a natural phenomenon or research finding. Include underlying mechanisms, experimental evidence, and real-world implications. Use precise scientific terminology.", 
                     "business"      : "Write a professional business analysis (500-1000 words) covering market trends, competitive landscape, or strategic insights. Use business terminology, data-driven arguments, and executive-level language.", 
                     "legal"         : "Draft a formal legal document excerpt (500-1000 words) such as a contract clause, terms of service, or policy statement. Use precise legal terminology, proper structure, and formal language appropriate for legal documents.",
                     "medical"       : "Write a clinical case description or medical research abstract (500-1000 words) with appropriate medical terminology. Include patient presentation, diagnostic approach, treatment, and outcomes or research methodology and findings.",
                     "journalism"    : "Write a journalistic news article (500-1000 words) in neutral reporting style. Include a compelling lead, factual reporting, quotes from sources, and balanced coverage. Follow AP style conventions.",
                     "marketing"     : "Write persuasive marketing content (500-1000 words) for a technology product or service. Include compelling value propositions, benefit-focused copy, clear calls to action, and engaging language that converts readers.", 
                     "social_media"  : "Write 5-7 engaging social media posts (500-1000 words total) discussing a technology trend. Use informal conversational tone, include relevant hashtags, emojis where appropriate, and encourage engagement. Mix different post types.",
                     "blog_personal" : "Write a personal blog post (500-1000 words) sharing personal experiences, opinions, or reflections on a topic. Use first-person perspective, informal conversational tone, and authentic voice.", 
                     "tutorial"      : "Write a comprehensive step-by-step tutorial (500-1000 words) teaching beginners how to accomplish a specific technical task. Use clear numbered steps, explanatory notes, common pitfalls, and helpful tips.",
                    }


def generate_with_ollama(prompt: str, domain: str, max_tokens: int = 600) -> str:
    """
    Generate text using Ollama with domain-specific parameters
    
    Arguments:
    ----------
        prompt     { str } : Generation prompt

        domain     { str } : Domain name for parameter lookup

        max_tokens { int } : Maximum tokens to generate
        
    Returns:
    --------
            { str }        : Generated text (empty string if failed)
    """
    url     = "http://localhost:11434/api/generate"
    params  = GENERATION_PARAMS.get(domain, GENERATION_PARAMS["default"])
    
    payload = {"model"   : MODEL_NAME,
               "prompt"  : prompt,
               "stream"  : False,
               "options" : {"temperature" : params["temperature"],
                            "top_p"       : params["top_p"],
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
            logger.warning(f"Ollama error: {response.status_code}")
            return ""

    except Exception as e:
        logger.warning(f"Request failed: {e}")
        return ""


def validate_generated_text(text: str, min_words: int = 100) -> bool:
    """
    Validate generated text quality
    
    Arguments:
    ----------
        text      { str } : Generated text

        min_words { int } : Minimum word count
        
    Returns:
    --------
           { bool }       : True if valid
    """
    if not text:
        return False
    
    word_count = len(text.split())
    
    return (word_count >= min_words)


def main():
    """
    Generate AI texts for all domains
    """
    print("=" * 70)
    print("TEXT-AUTH: Generating AI Data")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Generation strategy: Domain-specific prompts with adaptive parameters\n")
    
    total_generated = 0
    total_failed    = 0

    for domain_dir in HUMAN_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain = domain_dir.name
        
        if domain not in PROMPTS:
            logger.warning(f"Skipping {domain}: no prompt defined")
            continue

        (AI_DIR / domain).mkdir(parents = True, exist_ok = True)

        files           = list(domain_dir.glob("*.txt"))
        domain_success  = 0
        domain_failed   = 0

        logger.info(f"\n→ Generating for {domain} ({len(files)} samples)...")

        for i, file in enumerate(tqdm(files, desc = f"  {domain}")):
            prompt  = PROMPTS[domain]
            ai_text = generate_with_ollama(prompt  = prompt, 
                                           domain  = domain,
                                          )
            
            # Validate generated text
            if validate_generated_text(ai_text, min_words = 100):
                output_path = AI_DIR / domain / f"ai_{domain}_{i}.txt"
                
                with open(output_path, "w", encoding = "utf-8") as f:
                    f.write(ai_text)
                
                domain_success  += 1
                total_generated += 1
            
            else:
                logger.warning(f"  Failed to generate valid text for {domain}_{i} (skipping)")
                domain_failed += 1
                total_failed  += 1
        
        logger.info(f"  {domain}: {domain_success} generated, {domain_failed} failed")
    
    # Summary
    print("\n" + "=" * 70)
    print("Generation Summary")
    print("=" * 70)
    print(f"Total generated: {total_generated}")
    print(f"Total failed: {total_failed}")
    print("=" * 70)


# Execution
if __name__ == "__main__":
    main()