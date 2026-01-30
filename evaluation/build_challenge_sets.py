# DEPENDENCIES
import re
import json
import random
import requests
from tqdm import tqdm
from loguru import logger
from pathlib import Path


# Configuration
ADV_DIR = Path("evaluation/adversarial")
AI_DIR  = Path("evaluation/ai_generated")

# Ensure adversarial dirs exist
(ADV_DIR / "paraphrased").mkdir(parents = True, exist_ok = True)
(ADV_DIR / "cross_model").mkdir(parents = True, exist_ok = True)


def paraphrase_with_ollama(text: str, model: str = "mistral:7b") -> str:
    """
    Use Ollama to aggressively rephrase text
    
    Arguments:
    ----------
        text  { str } : Original text

        model { str } : Ollama model name
        
    Returns:
    --------
         { str }      : Paraphrased text
    """
    url     = "http://localhost:11434/api/generate"
    
    prompt  = f"""
                  You are an expert paraphraser. Completely rewrite the following text while preserving ONLY the core facts and meaning.

                  REQUIREMENTS:
                  1. Change ALL sentence structures - do not preserve any original phrasing
                  2. Use entirely different vocabulary (synonyms, alternative expressions)
                  3. Reorganize paragraphs and information flow completely
                  4. Vary sentence lengths dramatically from the original
                  5. Change passive/active voice throughout
                  6. Add transitional phrases in different locations
                  7. The rewritten text should be unrecognizable as a paraphrase to automated detectors

                  TARGET LENGTH: {len(text.split())} words (±20%)

                  ORIGINAL TEXT:
                  {text}

                  COMPLETELY REWRITTEN VERSION (do not include any preamble or meta-commentary):
               """
    
    payload = {"model"   : model,
               "prompt"  : prompt,
               "stream"  : False,
               "options" : {"temperature" : 0.8,
                            "num_predict" : min(len(text.split()) + 150, 900),
                           }
              }

    try:
        response = requests.post(url     = url, 
                                 json    = payload, 
                                 timeout = 90,
                                )

        if (response.status_code == 200):
            result = response.json().get("response", "").strip()
            
            # Validate paraphrase quality: at least 60% of original length
            if (len(result.split()) >= len(text.split()) * 0.6):
                return result
            
            else:
                logger.warning(f"Paraphrase too short ({len(result.split())} vs {len(text.split())} words), using original")
                return text
                
    except Exception as e:
        logger.warning(f"Paraphrase failed: {e}")
    
    # Fallback to original
    return text


def build_paraphrased(target_count: int = 500):
    """
    Build paraphrased challenge set
    
    Arguments:
    ----------
        target_count { int } : Number of paraphrased samples to generate
    """
    logger.info(f"\n→ Building paraphrased set (target: {target_count} samples)...")
    
    count           = 0
    processed_files = list()
    
    # Collect all AI-generated files
    for domain_dir in AI_DIR.iterdir():
        if domain_dir.is_dir():
            processed_files.extend(list(domain_dir.glob("*.txt")))
    
    random.shuffle(processed_files)
    
    # Progress bar
    pbar = tqdm(total = target_count, desc = "  Paraphrasing", unit = "sample")
    
    for file in processed_files:
        if (count >= target_count):
            break
        
        try:
            with open(file, encoding = "utf-8") as f:
                text = f.read()
            
            # Skip if too short or too long
            word_count = len(text.split())

            if ((word_count < 100) or (word_count > 2000)):
                continue
            
            # Paraphrase
            paraphrased = paraphrase_with_ollama(text)
            
            # Save with original filename
            output_path = ADV_DIR / "paraphrased" / file.name

            with open(output_path, "w", encoding = "utf-8") as f:
                f.write(paraphrased)
            
            count += 1
            pbar.update(1)
            
        except Exception as e:
            logger.warning(f"Error processing {file.name}: {e}")
    
    pbar.close()
    logger.info(f"  Generated {count} paraphrased samples\n")


def generate_cross_model_text(domain: str, model: str = "llama3:8b") -> str:
    """
    Generate text using different model than primary AI set
    
    Arguments:
    ----------
        domain { str } : Domain name

        model  { str } : Ollama model name
        
    Returns:
    --------
          { str }      : Generated text
    """
    url        = "http://localhost:11434/api/generate"
    
    # Cross-model prompts with different phrasing than primary generation
    prompt_map = {"general"       : "Compose a comprehensive informative article (300-500 words) explaining a topic of general interest.",   
                  "academic"      : "Compose a peer-reviewed research summary (250-400 words) discussing recent findings in a scientific field. Use formal scholarly language.",  
                  "creative"      : "Craft an engaging narrative passage (300-500 words) that transports readers into a vivid scene with rich sensory details and emotional depth.",
                  "ai_ml"         : "Explain a machine learning technique or AI concept (300-500 words) with technical depth appropriate for practitioners.",
                  "software_dev"  : "Create developer-focused documentation (300-500 words) for a software architecture or coding practice with practical examples.",
                  "technical_doc" : "Produce technical documentation (300-500 words) describing a system or API with specifications and usage guidelines.",
                  "engineering"   : "Compose an engineering analysis (300-500 words) evaluating a technical system design with performance considerations.",
                  "science"       : "Describe a scientific concept or research finding (300-500 words) with underlying mechanisms and evidence.",
                  "business"      : "Draft a business analysis report (300-500 words) examining market dynamics or strategic opportunities.",
                  "legal"         : "Compose a legal document section (300-500 words) using formal legal language and proper structure.",
                  "medical"       : "Write a medical case report or clinical abstract (300-500 words) with appropriate medical terminology.",
                  "journalism"    : "Report on a current event (300-500 words) using neutral journalistic style with factual coverage.",
                  "marketing"     : "Create marketing content (300-500 words) with persuasive language and benefit-driven messaging.",
                  "social_media"  : "Produce social media content (300-500 words) with casual tone and engaging language.",
                  "blog_personal" : "Compose a personal blog entry (300-500 words) sharing perspectives and experiences authentically.",
                  "tutorial"      : "Develop an instructional guide (300-500 words) with clear step-by-step directions for learners.",
                 }
    
    prompt     = prompt_map.get(domain, f"Write a well-structured {domain.replace('_', ' ')} text of 300-500 words.")
    
    payload    = {"model"   : model,
                  "prompt"  : prompt,
                  "stream"  : False,
                  "options" : {"temperature" : 0.8,
                               "num_predict" : 700,
                              }
                 }
    
    try:
        response = requests.post(url     = url, 
                                 json    = payload, 
                                 timeout = 120,
                                )
        
        if (response.status_code == 200):
            return response.json().get("response", "").strip()
            
    except Exception as e:
        logger.warning(f"Failed to generate for {domain}: {e}")
    
    return ""


def build_cross_model(target_count: int = 700):
    """
    Build cross-model challenge set
    
    Arguments:
    ----------
        target_count { int } : Number of cross-model samples to generate
    """
    logger.info(f"\n→ Building cross-model set (target: {target_count} samples)...")
    
    # All 16 domains
    domains = ["general", 
               "academic", 
               "creative", 
               "ai_ml", 
               "software_dev",
               "technical_doc", 
               "engineering", 
               "science", 
               "business", 
               "legal",
               "medical", 
               "journalism", 
               "marketing", 
               "social_media",
               "blog_personal", 
               "tutorial",
              ]
    
    samples_per_domain = target_count // len(domains)

    logger.info(f"  Target per domain: {samples_per_domain}")
    
    total_count        = 0
    
    for domain in domains:
        domain_count = 0
        attempts     = 0
        max_attempts = samples_per_domain + 25
        
        pbar         = tqdm(total = samples_per_domain, desc = f"  {domain}", leave = False)
        
        while ((domain_count < samples_per_domain) and (attempts < max_attempts)):
            attempts += 1
            
            text      = generate_cross_model_text(domain)
            
            # Validate generated text
            if (text and (len(text.split()) >= 100)):
                filepath = ADV_DIR / "cross_model" / f"llama3_{domain}_{domain_count}.txt"
                
                try:
                    with open(filepath, "w", encoding = "utf-8") as f:
                        f.write(text)
                    
                    domain_count += 1
                    total_count  += 1

                    pbar.update(1)
                    
                except Exception as e:
                    logger.warning(f"Save error: {e}")
        
        pbar.close()
        logger.info(f"  {domain}: {domain_count} samples")
    
    logger.info(f"\n  Total generated: {total_count} samples\n")


def validate_challenge_sets():
    """
    Validate generated challenge sets
    """
    logger.info("→ Validating challenge sets...")
    
    for subset in ["paraphrased", "cross_model"]:
        subset_path = ADV_DIR / subset
        
        if not subset_path.exists():
            logger.warning(f"  {subset} directory not found")
            continue
        
        files = list(subset_path.glob("*.txt"))
        
        if not files:
            logger.warning(f"  No files in {subset}")
            continue
        
        # Check sample
        sample_file = random.choice(files)

        with open(sample_file) as f:
            sample_text = f.read()
        
        word_count = len(sample_text.split())
        
        logger.info(f"  {subset}: {len(files)} files")
        logger.info(f"    Sample length: {word_count} words")
        logger.info(f"    Sample file: {sample_file.name[:50]}...")


def main():
    """
    Main execution
    """
    print("=" * 70)
    print("TEXT-AUTH: Building Challenge Sets")
    print("=" * 70)
    
    print("\nEnsure Ollama is running with models:")
    print("  - mistral:7b (for paraphrasing)")
    print("  - llama3:8b (for cross-model generation)")
    print()
    
    # Build paraphrased set (500 samples) - UNCOMMENTED
    build_paraphrased(target_count = 500)
    
    # Build cross-model set (700 samples)
    build_cross_model(target_count = 700)
    
    # Validate
    validate_challenge_sets()
    
    print("\n" + "=" * 70)
    print("Challenge Sets Ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()