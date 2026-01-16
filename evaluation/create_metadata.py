# DEPENDENCIES
import json
from pathlib import Path


# Count actual samples
def count_samples(dir_path: Path) -> int:
    total = 0

    if dir_path.exists():
        for domain_dir in dir_path.iterdir():
            if domain_dir.is_dir():
                total += len(list(domain_dir.glob("*.txt")))
    
    return total


human_count       = count_samples(Path("evaluation/human"))
ai_count          = count_samples(Path("evaluation/ai_generated"))
paraphrased_count = count_samples(Path("evaluation/adversarial/paraphrased")) 
cross_model_count = count_samples(Path("evaluation/adversarial/cross_model"))

metadata          = {"dataset_name"      : "TEXT-AUTH-Eval",
                     "version"           : "1.0",
                     "total_samples"     : human_count + ai_count + paraphrased_count + cross_model_count,
                     "human_samples"     : human_count,
                     "ai_samples"        : ai_count,
                     "challenge_samples" : {"paraphrased" : paraphrased_count,
                                            "cross_model" : cross_model_count,
                                           },
                     "domains"           : ["general", "academic", "creative", "ai_ml", "software_dev", "technical_doc", "engineering", "science", "business", "legal", "medical", "journalism", "marketing", "social_media", "blog_personal", "tutorial"],
                     "human_sources"     : {"general"       : "Wikipedia",
                                            "academic"      : "scientific_papers (arXiv abstracts)",
                                            "creative"      : "Project Gutenberg / C4 filtered",
                                            "ai_ml"         : "scientific_papers (arXiv with ML keywords)",
                                            "software_dev"  : "C4 filtered (code/documentation keywords)",
                                            "technical_doc" : "C4 filtered (documentation keywords)",
                                            "engineering"   : "scientific_papers (arXiv engineering)",
                                            "science"       : "C4 filtered (scientific keywords)",
                                            "business"      : "C4 filtered (business/financial keywords)",
                                            "legal"         : "lex_glue / C4 filtered (legal keywords)",
                                            "medical"       : "scientific_papers (PubMed abstracts)",
                                            "journalism"    : "cnn_dailymail",
                                            "marketing"     : "C4 filtered (marketing keywords)",
                                            "social_media"  : "tweet_eval / C4 filtered (social keywords)",
                                            "blog_personal" : "C4 filtered (personal narrative keywords)",
                                            "tutorial"      : "C4 filtered (tutorial/guide keywords)",
                                           },
                     "ai_generation"     : {"primary_model" : "mistral:7b (via Ollama)",
                                            "cross_model"   : "llama3:8b (via Ollama)",
                                            "paraphrasing"  : "mistral:7b (via Ollama instruction-based rephrasing)",
                                           },
                     "notes"             : ["All AI-generated texts produced using local Ollama models to avoid Hugging Face downloads",
                                            "Paraphrased set created by instructing mistral:7b to rephrase original AI texts",
                                            "Cross-model set generated using llama3:8b (unseen during primary AI generation)",
                                            "Human texts sourced exclusively from public, auto-downloadable datasets",
                                           ],
                     "license"           : "CC BY / Public Domain / Fair Use — for research only",
                     "created"           : "2025",
                     "compatible_with"   : "TEXT-AUTH v1.0.0",
                    }


# Save to evaluation directory
output_path = Path("evaluation/metadata.json")

with open(output_path, "w") as f:
    json.dump(obj    = metadata, 
              fp     = f, 
              indent = 4,
             )


print(f"metadata.json saved to {output_path}")
print(f"Dataset Summary:")
print(f"   Human:       {human_count}")
print(f"   AI:          {ai_count}")
print(f"   Paraphrased: {paraphrased_count}")
print(f"   Cross-Model: {cross_model_count}")
print(f"   TOTAL:       {human_count + ai_count + paraphrased_count + cross_model_count}")
