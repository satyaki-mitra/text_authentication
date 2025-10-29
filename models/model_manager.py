# DEPENDENCIES
import os
import gc
import json
import torch
import spacy
import threading
import subprocess
from typing import Any
from typing import Dict
from typing import Union
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime
from transformers import pipeline
from collections import OrderedDict
from config.settings import settings
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from config.model_config import ModelType
from config.model_config import ModelConfig
from transformers import AutoModelForMaskedLM
from config.model_config import MODEL_REGISTRY
from config.model_config import get_model_config
from config.model_config import get_required_models
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification


class ModelCache:
    """
    LRU cache for models with size limit
    """
    def __init__(self, max_size: int = 5):
        self.max_size            = max_size
        self.cache : OrderedDict = OrderedDict()
        self.lock                = threading.Lock()
    

    def get(self, key: str) -> Optional[Any]:
        """
        Get model from cache
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                logger.debug(f"Cache hit for model: {key}")
                
                return self.cache[key]
            
            logger.debug(f"Cache miss for model: {key}")
            
            return None
    

    def put(self, key: str, model: Any):
        """
        Add model to cache
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            
            else:
                if (len(self.cache) >= self.max_size):
                    # Remove least recently used
                    removed_key   = next(iter(self.cache))
                    removed_model = self.cache.pop(removed_key)
                    
                    # Clean up memory
                    if hasattr(removed_model, 'to'):
                        removed_model.to('cpu')
                    
                    del removed_model
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    logger.info(f"Evicted model from cache: {removed_key}")
                
                self.cache[key] = model

                logger.info(f"Added model to cache: {key}")
    

    def clear(self):
        """
        Clear all cached models
        """
        with self.lock:
            for model in self.cache.values():
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            
            self.cache.clear()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleared model cache")
    

    def size(self) -> int:
        """
        Get current cache size
        """
        return len(self.cache)



class ModelManager:
    """
    Central model management system
    """
    def __init__(self):
        self.cache         = ModelCache(max_size = settings.MAX_CACHED_MODELS)
        self.device        = torch.device(settings.DEVICE if torch.cuda.is_available() else "cpu")
        self.cache_dir     = settings.MODEL_CACHE_DIR

        self.cache_dir.mkdir(parents  = True, 
                             exist_ok = True,
                            )
        
        # Model metadata tracking
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata      = self._load_metadata()
        
        logger.info(f"ModelManager initialized with device: {self.device}")
        logger.info(f"Model cache directory: {self.cache_dir}")
    

    def _load_metadata(self) -> Dict:
        """
        Load model metadata from disk
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            
            except Exception as e:
                logger.warning(f"Failed to load metadata: {repr(e)}")
        
        return {}
    

    def _save_metadata(self):
        """
        Save model metadata to disk
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(obj    = self.metadata, 
                          fp     = f, 
                          indent = 4,
                         )
        
        except Exception as e:
            logger.error(f"Failed to save metadata: {repr(e)}")
    

    def _update_metadata(self, model_name: str, model_config: ModelConfig):
        """
        Update metadata for a model
        """
        self.metadata[model_name] = {"model_id"      : model_config.model_id,
                                     "model_type"    : model_config.model_type.value,
                                     "downloaded_at" : datetime.now().isoformat(),
                                     "size_mb"       : model_config.size_mb,
                                     "last_used"     : datetime.now().isoformat(),
                                    }
        self._save_metadata()
    

    def is_model_downloaded(self, model_name: str) -> bool:
        """
        Check if model is already downloaded
        """
        model_config = get_model_config(model_name = model_name)
        
        if not model_config:
            return False
        
        # Check if model exists in cache directory
        model_path = self.cache_dir / model_config.model_id.replace("/", "_")
        
        return model_path.exists() and model_name in self.metadata
    

    def load_model(self, model_name: str, force_download: bool = False) -> Any:
        """
        Load a model by name
        
        Arguments:
        ----------
            model_name     { str }  : Name from MODEL_REGISTRY

            force_download { bool } : Force re-download even if cached
            
        Returns:
        --------
                  { Any }           : Model instance
        """
        # Check cache first
        if not force_download:
            cached = self.cache.get(key = model_name)
            
            if cached is not None:
                return cached
        
        # Get model configuration
        model_config = get_model_config(model_name = model_name)
        
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Loading model: {model_name} ({model_config.model_id})")
        
        try:
            # Load based on model type
            if (model_config.model_type == ModelType.SENTENCE_TRANSFORMER):
                model = self._load_sentence_transformer(config = model_config)

            elif (model_config.model_type == ModelType.GPT):
                model = self._load_gpt_model(config = model_config)

            elif (model_config.model_type == ModelType.CLASSIFIER):
                model = self._load_classifier(config = model_config)

            elif (model_config.model_type == ModelType.SEQUENCE_CLASSIFICATION):
                model = self._load_sequence_classifier(config = model_config)

            elif (model_config.model_type == ModelType.TRANSFORMER):
                model = self._load_transformer(config = model_config)

            elif (model_config.model_type == ModelType.RULE_BASED):
                # Check if it's a spaCy model
                if model_config.additional_params.get("is_spacy_model", False):
                    model = self._load_spacy_model(config = model_config)
                
                else:
                    raise ValueError(f"Unknown rule-based model type: {model_name}")

            else:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
            
            # Update metadata
            self._update_metadata(model_name   = model_name, 
                                  model_config = model_config,
                                 )
            
            # Cache the model
            if model_config.cache_model:
                self.cache.put(key   = model_name, 
                               model = model,
                              )
            
            logger.success(f"Successfully loaded model: {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {repr(e)}")
            raise

    
    def load_tokenizer(self, model_name: str) -> Any:
        """
        Load tokenizer for a model
        
        Arguments:
        ----------
            model_name { str } : Name from MODEL_REGISTRY
            
        Returns:
        --------
            { Any }            : Tokenizer instance
        """
        model_config = get_model_config(model_name = model_name)
        
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Loading tokenizer for: {model_name}")
        
        try:
            if (model_config.model_type in [ModelType.GPT, ModelType.CLASSIFIER, ModelType.SEQUENCE_CLASSIFICATION, ModelType.TRANSFORMER]):
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                                          cache_dir                     = str(self.cache_dir),
                                                         )
                
                logger.success(f"Successfully loaded tokenizer for: {model_name}")
                return tokenizer
            
            else:
                raise ValueError(f"Model type {model_config.model_type} doesn't require a separate tokenizer")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {repr(e)}")
            raise

    
    def _load_sentence_transformer(self, config: ModelConfig) -> SentenceTransformer:
        """
        Load SentenceTransformer model
        """
        model = SentenceTransformer(model_name_or_path = config.model_id,
                                    cache_folder       = str(self.cache_dir),
                                    device             = str(self.device),
                                   )
        return model
        
    
    def _load_gpt_model(self, config: ModelConfig) -> tuple:
        """
        Load GPT-style model with tokenizer
        """
        model     = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path = config.model_id,
                                                    cache_dir                     = str(self.cache_dir),
                                                   )

        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path = config.model_id,
                                                  cache_dir                     = str(self.cache_dir),
                                                 )
        
        # Move to device
        model     = model.to(self.device)

        model.eval()
        
        # Apply quantization if enabled
        if (settings.USE_QUANTIZATION and config.quantizable):
            model = self._quantize_model(model = model)
        
        return (model, tokenizer)

    
    def _load_classifier(self, config: ModelConfig) -> Any:
        """
        Load classification model (for zero-shot, etc.)
        """
        # For zero-shot classification models
        pipe = pipeline("zero-shot-classification",
                        model        = config.model_id,
                        device       = 0 if self.device.type == "cuda" else -1,
                        model_kwargs = {"cache_dir": str(self.cache_dir)},
                       )
        
        return pipe

    
    def _load_sequence_classifier(self, config: ModelConfig) -> Any:
        """
        Load sequence classification model (for domain classification)
        """
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = config.model_id,
                                                                   cache_dir                     = str(self.cache_dir),
                                                                   num_labels                    = config.additional_params.get('num_labels', 2),
                                                                  )
        
        # Move to device
        model = model.to(self.device)
        
        model.eval()
        
        # Apply quantization if enabled
        if (settings.USE_QUANTIZATION and config.quantizable):
            model = self._quantize_model(model = model)
        
        return model

    
    def _load_transformer(self, config: ModelConfig) -> tuple:
        """
        Load masking transformer model
        """
        model     = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path = config.model_id,
                                                         cache_dir                     = str(self.cache_dir),
                                                        )

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = config.model_id,
                                                  cache_dir                     = str(self.cache_dir),
                                                 )
        
        # Move to device
        model     = model.to(self.device)

        model.eval()
        
        # Apply quantization if enabled
        if (settings.USE_QUANTIZATION and config.quantizable):
            model = self._quantize_model(model)
        
        return (model, tokenizer)

    
    def _quantize_model(self, model):
        """
        Apply INT8 quantization to model
        """
        try:
            if hasattr(torch.quantization, 'quantize_dynamic'):
                quantized_model = torch.quantization.quantize_dynamic(model        = model,
                                                                      qconfig_spec = {torch.nn.Linear},
                                                                      dtype        = torch.qint8,
                                                                     )
                logger.info("Applied INT8 quantization to model")
                
                return quantized_model

        except Exception as e:
            logger.warning(f"Quantization failed: {repr(e)}, using original model")
    
        return model
    

    def load_pipeline(self, model_name: str, task: str) -> pipeline:
        """
        Load a Hugging Face pipeline
        """
        model_config = get_model_config(model_name = model_name)
        
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Loading pipeline: {task} with {model_name}")
        
        pipe = pipeline(task         = task,
                        model        = model_config.model_id,
                        device       = 0 if self.device.type == "cuda" else -1,
                        model_kwargs = {"cache_dir": str(self.cache_dir)},
                       )
        
        return pipe
    

    def _load_spacy_model(self, config: ModelConfig):
        """
        Load spaCy model
        """
        try:
            model = spacy.load(config.model_id)
            logger.info(f"Loaded spaCy model: {config.model_id}")
            
            return model
        
        except OSError:
            # Model not downloaded, install it
            logger.info(f"Downloading spaCy model: {config.model_id}")
            
            subprocess.run(["python", "-m", "spacy", "download", config.model_id], check = True)
            model = spacy.load(config.model_id)

            return model

    
    def download_model(self, model_name: str) -> bool:
        """
        Download model without loading it into memory
        
        Arguments:
        ----------
            model_name { str } : Name from MODEL_REGISTRY
            
        Returns:
        --------
               { bool }        : True if successful, False otherwise
        """
        model_config = get_model_config(model_name)
        
        if not model_config:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        if self.is_model_downloaded(model_name):
            logger.info(f"Model already downloaded: {model_name}")
            return True
        
        logger.info(f"Downloading model: {model_name} ({model_config.model_id})")
        
        try:
            if model_config.model_type == ModelType.SENTENCE_TRANSFORMER:
                SentenceTransformer(model_name_or_path = model_config.model_id,
                                    cache_folder       = str(self.cache_dir),
                                   )
                                   
            elif (model_config.model_type == ModelType.GPT):
                GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                                cache_dir                     = str(self.cache_dir),
                                               )

                GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                              cache_dir                     = str(self.cache_dir),
                                             )

            elif (model_config.model_type == ModelType.SEQUENCE_CLASSIFICATION):
                AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                                                   cache_dir                     = str(self.cache_dir),
                                                                  )

                AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                              cache_dir                     = str(self.cache_dir),
                                             )

            elif (model_config.model_type == ModelType.RULE_BASED):
                if model_config.additional_params.get("is_spacy_model", False):
                    subprocess.run(["python", "-m", "spacy", "download", model_config.model_id], check = True)

                else:
                    logger.warning(f"Cannot pre-download rule-based model: {model_name}")
                    # Mark as "downloaded"
                    return True  

            else:
                # Generic transformer models
                AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                                                   cache_dir                     = str(self.cache_dir),
                                                                  )

                AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_config.model_id,
                                              cache_dir                     = str(self.cache_dir),
                                             )
            
            self._update_metadata(model_name, model_config)
            
            logger.success(f"Successfully downloaded: {model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {repr(e)}")
            return False
    

    def download_all_required(self) -> Dict[str, bool]:
        """
        Download all required models
        
        Returns:
        --------
            { dict }    : Dict mapping model names to success status
        """
        required_models = get_required_models()
        results         = dict()
        
        logger.info(f"Downloading {len(required_models)} required models...")
        
        for model_name in required_models:
            results[model_name] = self.download_model(model_name = model_name)
        
        success_count = sum(1 for v in results.values() if v)

        logger.info(f"Downloaded {success_count}/{len(required_models)} required models")
        
        return results
        
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a model
        """
        return self.metadata.get(model_name)
    

    def list_downloaded_models(self) -> list:
        """
        List all downloaded models
        """
        return list(self.metadata.keys())
    

    def clear_cache(self):
        """
        Clear model cache
        """
        self.cache.clear()
        logger.info("Model cache cleared")
    

    def unload_model(self, model_name: str):
        """
        Unload a specific model from cache
        """
        with self.cache.lock:
            if model_name in self.cache.cache:
                model = self.cache.cache.pop(model_name)
                if hasattr(model, 'to'):
                    model.to('cpu')
                
                del model
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Unloaded model: {model_name}")
    

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        """
        stats = {"cached_models" : self.cache.size(),
                 "device"        : str(self.device),
                }
        
        if torch.cuda.is_available():
            stats.update({"gpu_allocated_mb"     : torch.cuda.memory_allocated() / 1024**2,
                          "gpu_reserved_mb"      : torch.cuda.memory_reserved() / 1024**2,
                          "gpu_max_allocated_mb" : torch.cuda.max_memory_allocated() / 1024**2,
                        })
        
        return stats

    
    def optimize_memory(self):
        """
        Optimize memory usage
        """
        logger.info("Optimizing memory...")
        
        # Clear unused cached models
        self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory optimization complete")
        logger.info(f"Memory usage: {self.get_memory_usage()}")



# Singleton instance
_model_manager_instance : Optional[ModelManager] = None
_manager_lock                                    = threading.Lock()


def get_model_manager() -> ModelManager:
    """
    Get singleton ModelManager instance
    """
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _manager_lock:
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager()
    
    return _model_manager_instance




# Export
__all__ = ["ModelManager",
           "ModelCache",
           "get_model_manager",
          ]