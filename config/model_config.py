# DEPENDENCIES
from typing import Any
from typing import Dict
from typing import Optional
from config.enums import ModelType
from config.schemas import ModelConfig


MODEL_REGISTRY : Dict[str, ModelConfig] = {"perplexity_reference_lm"    : ModelConfig(model_id          = "gpt2",
                                                                                      model_type        = ModelType.LANGUAGE_MODEL,
                                                                                      description       = "Reference language model for statistical perplexity estimation",
                                                                                      size_mb           = 548,
                                                                                      required          = True,
                                                                                      download_priority = 1,
                                                                                      max_length        = 1024,
                                                                                      batch_size        = 8,
                                                                                      quantizable       = True,
                                                                                     ), 
                                           "semantic_primary"           : ModelConfig(model_id          = "sentence-transformers/all-MiniLM-L6-v2",
                                                                                      model_type        = ModelType.SENTENCE_TRANSFORMER,
                                                                                      description       = "Lightweight semantic embeddings (80MB)",
                                                                                      size_mb           = 80,
                                                                                      required          = True,
                                                                                      download_priority = 1,
                                                                                      max_length        = 256,
                                                                                      batch_size        = 32,
                                                                                     ),
                                           "semantic_secondary"         : ModelConfig(model_id          = "sentence-transformers/all-mpnet-base-v2",
                                                                                      model_type        = ModelType.SENTENCE_TRANSFORMER,
                                                                                      description       = "Higher quality semantic embeddings (backup)",
                                                                                      size_mb           = 420,
                                                                                      required          = False,
                                                                                      download_priority = 3,
                                                                                      max_length        = 384,
                                                                                      batch_size        = 16,
                                                                                     ),
                                           "linguistic_spacy"           : ModelConfig(model_id          = "en_core_web_sm",
                                                                                      model_type        = ModelType.RULE_BASED,
                                                                                      description       = "spaCy small English model for POS tagging",
                                                                                      size_mb           = 13,
                                                                                      required          = True,
                                                                                      download_priority = 1,
                                                                                      batch_size        = 16,
                                                                                      additional_params = {"is_spacy_model": True},
                                                                                     ),
                                           "content_domain_classifier"  : ModelConfig(model_id          = "facebook/bart-large-mnli",
                                                                                      model_type        = ModelType.CLASSIFIER,
                                                                                      description       = "Zero-shot content domain inference model",
                                                                                      size_mb           = 500,
                                                                                      required          = True,  
                                                                                      download_priority = 1,     
                                                                                      max_length        = 512,
                                                                                      batch_size        = 8,
                                                                                      quantizable       = True,
                                                                                     ),
                                           "domain_classifier_fallback" : ModelConfig(model_id          = "microsoft/deberta-v3-small",
                                                                                      model_type        = ModelType.CLASSIFIER,
                                                                                      description       = "Fast fallback zero-shot classifier (DeBERTa-small)",
                                                                                      size_mb           = 240,
                                                                                      required          = True,
                                                                                      download_priority = 2,
                                                                                      max_length        = 512,
                                                                                      batch_size        = 16,
                                                                                      quantizable       = True,
                                                                                     ),
                                           "multi_perturbation_base"    : ModelConfig(model_id          = "gpt2",
                                                                                      model_type        = ModelType.CAUSAL_LM,
                                                                                      description       = "MultiPerturbationStability model (reuses gpt2)",
                                                                                      size_mb           = 0,  
                                                                                      required          = True,  
                                                                                      download_priority = 4,
                                                                                      max_length        = 1024,
                                                                                      batch_size        = 4,
                                                                                     ),
                                           "multi_perturbation_mask"    : ModelConfig(model_id          = "distilroberta-base",
                                                                                      model_type        = ModelType.MASKED_LM,
                                                                                      description       = "Masked LM for text perturbation",
                                                                                      size_mb           = 330,
                                                                                      required          = True, 
                                                                                      download_priority = 4,
                                                                                      max_length        = 512,
                                                                                      batch_size        = 8,
                                                                                     ),
                                           "language_detector"          : ModelConfig(model_id          = "papluca/xlm-roberta-base-language-detection",
                                                                                      model_type        = ModelType.CLASSIFIER,
                                                                                      description       = "Language detection for routing; not used in authenticity scoring",
                                                                                      size_mb           = 1100,
                                                                                      required          = False,
                                                                                      download_priority = 5,
                                                                                      max_length        = 512,
                                                                                      batch_size        = 16,
                                                                                     ),
                                          }


# MODEL GROUPS FOR BATCH DOWNLOADING 
MODEL_GROUPS                            = {"minimal"   : ["perplexity_reference_lm", "content_domain_classifier"],
                                           "essential" : ["perplexity_reference_lm", "semantic_primary", "linguistic_spacy", "content_domain_classifier"],
                                           "extended"  : ["semantic_secondary", "multi_perturbation_mask", "domain_classifier_fallback"],
                                           "optional"  : ["language_detector"],
                                          }


# MODEL WEIGHTS FOR ENSEMBLE : For 6 metrics implemented
DEFAULT_MODEL_WEIGHTS                   = {"structural"                   : 0.20,  # No model needed
                                           "perplexity"                   : 0.20,  # reference language model
                                           "entropy"                      : 0.15,  # token distribution statistics
                                           "semantic"                     : 0.20,  # all-MiniLM-L6-v2
                                           "linguistic"                   : 0.15,  # spacy
                                           "multi_perturbation_stability" : 0.10,  # gpt2 + distilroberta (optional)
                                          }


# HELPER FUNCTIONS
def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model
    """
    return MODEL_REGISTRY.get(model_name)


def get_required_models() -> Dict[str, ModelConfig]:
    """
    Get all required models
    """
    return {name: config for name, config in MODEL_REGISTRY.items() if config.required}


def get_models_by_priority(priority: int) -> Dict[str, ModelConfig]:
    """
    Get models by download priority
    """
    return {name: config for name, config in MODEL_REGISTRY.items() if config.download_priority == priority}


def get_models_by_group(group_name: str) -> Dict[str, ModelConfig]:
    """
    Get models belonging to a specific group
    """
    if group_name not in MODEL_GROUPS:
        return {}
    
    model_names = MODEL_GROUPS[group_name]
    return {name: MODEL_REGISTRY[name] for name in model_names if name in MODEL_REGISTRY}


def get_total_size_mb(group_name: Optional[str] = None) -> int:
    """
    Calculate total size of models
    
    Arguments:
    ----------
        group_name : If specified, only count models in that group
    
    Returns:
    --------
        Total size in MB
    """
    if group_name:
        models = get_models_by_group(group_name)
    
    else:
        models = MODEL_REGISTRY
    
    return sum(config.size_mb for config in models.values())


def get_required_size_mb() -> int:
    """
    Calculate total size of required models only
    """
    return sum(config.size_mb for config in MODEL_REGISTRY.values() if config.required)


def print_model_summary():
    """
    Print a summary of models and their sizes
    """
    print("\n" + "="*70)
    print("MODEL REGISTRY SUMMARY")
    print("="*70)
    
    for group_name, model_names in MODEL_GROUPS.items():
        group_size = get_total_size_mb(group_name)
        print(f"\n[{group_name.upper()}] - Total: {group_size} MB")
        print("-" * 70)
        
        for model_name in model_names:
            if model_name in MODEL_REGISTRY:
                config = MODEL_REGISTRY[model_name]
                req_str = "✓ REQUIRED" if config.required else "  optional"
                print(f"  {req_str} | {model_name:30s} | {config.size_mb:5d} MB | {config.model_id}")
    
    print("\n" + "="*70)
    print(f"TOTAL REQUIRED MODELS: {get_required_size_mb()} MB")
    print(f"TOTAL ALL MODELS: {get_total_size_mb()} MB")
    print("="*70 + "\n")


# SPACY MODEL INSTALLATION

def get_spacy_download_commands() -> list:
    """
    Get commands to download spaCy models
    """
    spacy_models = [config for config in MODEL_REGISTRY.values() if config.additional_params.get("is_spacy_model", False)]
    
    commands     = list()

    for config in spacy_models:
        commands.append(f"python -m spacy download {config.model_id}")
    
    return commands


# Export
__all__ = ["ModelType",
           "ModelConfig",
           "MODEL_GROUPS",
           "MODEL_REGISTRY",
           "get_model_config",
           "get_total_size_mb",
           "get_required_models",
           "get_models_by_group",
           "print_model_summary",
           "get_required_size_mb",
           "DEFAULT_MODEL_WEIGHTS",
            "get_models_by_priority",
           "get_spacy_download_commands",
          ]


# AUTO-RUN SUMMARY 
if __name__ == "__main__":

    print_model_summary()
    
    print("\nSPACY MODEL INSTALLATION:")

    print("-" * 70)
    for cmd in get_spacy_download_commands():
        print(f"  {cmd}")

    print()