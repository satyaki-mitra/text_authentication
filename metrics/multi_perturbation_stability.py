# DEPENDENCIES
import re
import torch
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from transformers import pipeline
from config.threshold_config import Domain
from metrics.base_metric import BaseMetric
from metrics.base_metric import MetricResult
from models.model_manager import get_model_manager
from config.threshold_config import get_threshold_for_domain



class MultiPerturbationStabilityMetric(BaseMetric):
    """
    Multi-Perturbation Stability Metric (MPSM) 
    
    A hybrid approach for combining multiple perturbation techniques for robust AI-generated text detection
    
    Measures:
    - Text stability under random perturbations
    - Likelihood curvature analysis
    - Masked token prediction analysis

    Perturbation Methods:
    - Word deletation & swapping
    - RoBERTa mask filling
    - Synonym replacement
    - Chunk-based stability Analysis
    """
    def __init__(self):
        super().__init__(name        = "multi_perturbation_stability",
                         description = "Text stability analysis under multi-perturbations techniques",
                        )
        
        self.gpt_model      = None
        self.gpt_tokenizer  = None
        self.mask_model     = None
        self.mask_tokenizer = None
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    

    def initialize(self) -> bool:
        """
        Initialize the MultiPerturbationStability metric
        """
        try:
            logger.info("Initializing MultiPerturbationStability metric...")
            
            # Load GPT-2 model for likelihood calculation
            model_manager = get_model_manager()
            gpt_result    = model_manager.load_model(model_name = "multi_perturbation_base")
            
            if isinstance(gpt_result, tuple):
                self.gpt_model, self.gpt_tokenizer = gpt_result
                # Move model to appropriate device
                self.gpt_model.to(self.device)
                logger.success("✓ GPT-2 model loaded for MultiPerturbationStability")
           
            else:
                logger.error("Failed to load GPT-2 model for MultiPerturbationStability")
                return False
            
            # Load masked language model for perturbations
            mask_result = model_manager.load_model("multi_perturbation_mask")
            
            if (isinstance(mask_result, tuple)):
                self.mask_model, self.mask_tokenizer = mask_result
                # Move model to appropriate device
                self.mask_model.to(self.device)
                
                # Ensure tokenizer has padding token
                if (self.mask_tokenizer.pad_token is None):
                    self.mask_tokenizer.pad_token = self.mask_tokenizer.eos_token or '[PAD]'

                # Ensure tokenizer has mask token
                if not hasattr(self.mask_tokenizer, 'mask_token') or self.mask_tokenizer.mask_token is None:
                    self.mask_tokenizer.mask_token = "<mask>"
                
                logger.success("✓ DistilRoBERTa model loaded for MultiPerturbationStability")

            else:
                logger.warning("Failed to load mask model, using GPT-2 only")
            
            # Verify model loading
            if not self._verify_model_loading():
                logger.error("Model verification failed")
                return False
            
            self.is_initialized = True
            
            logger.success("MultiPerturbationStability metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MultiPerturbationStability metric: {repr(e)}")
            return False
    

    def _verify_model_loading(self) -> bool:
        """
        Verify that models are properly loaded and working
        """
        try:
            test_text = "This is a test sentence for model verification."
            
            # Test GPT-2 model
            if self.gpt_model and self.gpt_tokenizer:
                gpt_likelihood = self._calculate_likelihood(text = test_text)
                logger.info(f"GPT-2 test - Likelihood: {gpt_likelihood:.4f}")
            
            else:
                logger.error("GPT-2 model not loaded")
                return False
            
            # Test DistilRoBERTa model if available
            if self.mask_model and self.mask_tokenizer:
                # Test mask token
                if hasattr(self.mask_tokenizer, 'mask_token') and self.mask_tokenizer.mask_token:
                    logger.info(f"DistilRoBERTa mask token: '{self.mask_tokenizer.mask_token}'")
                    
                    # Test basic tokenization
                    inputs = self.mask_tokenizer(test_text, return_tensors = "pt")
                    logger.info(f"DistilRoBERTa tokenization test - Input shape: {inputs['input_ids'].shape}")
                
                else:
                    logger.warning("DistilRoBERTa mask token not available")
            
            else:
                logger.warning("DistilRoBERTa model not loaded")
            
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute MultiPerturbationStability analysis with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if ((not text) or (len(text.strip()) < 50)):
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for MultiPerturbationStability analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                                  = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                       = get_threshold_for_domain(domain)
            multi_perturbation_stability_thresholds = domain_thresholds.multi_perturbation_stability
            
            # Check if we should run this computationally expensive metric
            if (kwargs.get('skip_expensive', False)):
                logger.info("Skipping MultiPerturbationStability due to computational constraints")
                
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.3,
                                    error             = "Skipped for performance",
                                   )
            
            # Calculate MultiPerturbationStability features
            features                        = self._calculate_stability_features(text = text)
            
            # Calculate raw MultiPerturbationStability score (0-1 scale)
            raw_stability_score, confidence = self._analyze_stability_patterns(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob = self._apply_domain_thresholds(raw_score  = raw_stability_score, 
                                                                            thresholds = multi_perturbation_stability_thresholds, 
                                                                            features   = features,
                                                                           )
            
            # Apply confidence multiplier from domain thresholds
            confidence                     *= multi_perturbation_stability_thresholds.confidence_multiplier
            confidence                      = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : multi_perturbation_stability_thresholds.ai_threshold,
                                                     'human_threshold' : multi_perturbation_stability_thresholds.human_threshold,
                                                     'raw_score'       : raw_stability_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in MultiPerturbationStability computation: {repr(e)}")

            return MetricResult(metric_name       = self.name,
                                ai_probability    = 0.5,
                                human_probability = 0.5,
                                mixed_probability = 0.0,
                                confidence        = 0.0,
                                error             = str(e),
                               )
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        ai_threshold    = thresholds.ai_threshold      # e.g., 0.75 for GENERAL, 0.80 for ACADEMIC
        human_threshold = thresholds.human_threshold   # e.g., 0.25 for GENERAL, 0.20 for ACADEMIC
        
        # Calculate probabilities based on threshold distances
        if (raw_score >= ai_threshold):
            # Above AI threshold - strongly AI
            distance_from_threshold = raw_score - ai_threshold
            ai_prob                 = 0.7 + (distance_from_threshold * 0.3)  # 0.7 to 1.0
            human_prob              = 0.3 - (distance_from_threshold * 0.3)  # 0.3 to 0.0

        elif (raw_score <= human_threshold):
            # Below human threshold - strongly human
            distance_from_threshold = human_threshold - raw_score
            ai_prob                 = 0.3 - (distance_from_threshold * 0.3)  # 0.3 to 0.0
            human_prob              = 0.7 + (distance_from_threshold * 0.3)  # 0.7 to 1.0

        else:
            # Between thresholds - uncertain zone
            range_width             = ai_threshold - human_threshold

            if (range_width > 0):
                position_in_range = (raw_score - human_threshold) / range_width
                ai_prob           = 0.3 + (position_in_range * 0.4)  # 0.3 to 0.7
                human_prob        = 0.7 - (position_in_range * 0.4)  # 0.7 to 0.3
            
            else:
                ai_prob    = 0.5
                human_prob = 0.5
        
        # Ensure probabilities are valid
        ai_prob    = max(0.0, min(1.0, ai_prob))
        human_prob = max(0.0, min(1.0, human_prob))
        
        # Calculate mixed probability based on stability variance
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob

        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob
    

    def _calculate_stability_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive MultiPerturbationStability features with diagnostic logging
        """
        if not self.gpt_model or not self.gpt_tokenizer:
            return self._get_default_features()
        
        try:
            # Preprocess text for better analysis
            processed_text        = self._preprocess_text_for_analysis(text = text)
            
            # Calculate original text likelihood
            original_likelihood   = self._calculate_likelihood(text = processed_text)
            logger.debug(f"Original likelihood: {original_likelihood:.4f}")
            
            # Generate perturbations and calculate perturbed likelihoods
            perturbations         = self._generate_perturbations(text              = processed_text,
                                                                 num_perturbations = 10,
                                                                )
            logger.debug(f"Generated {len(perturbations)} perturbations")

            perturbed_likelihoods = list()
            
            for idx, perturbed_text in enumerate(perturbations):
                if (perturbed_text and (perturbed_text != processed_text)):
                    likelihood = self._calculate_likelihood(text = perturbed_text)
                    
                    if (likelihood > 0):
                        perturbed_likelihoods.append(likelihood)
                        logger.debug(f"Perturbation {idx}: likelihood={likelihood:.4f}")
            
            logger.info(f"Valid perturbations: {len(perturbed_likelihoods)}/{len(perturbations)}")
            
            # Calculate stability metrics
            if perturbed_likelihoods:
                stability_score          = self._calculate_stability_score(original_likelihood   = original_likelihood, 
                                                                           perturbed_likelihoods = perturbed_likelihoods,
                                                                          )

                curvature_score          = self._calculate_curvature_score(original_likelihood   = original_likelihood, 
                                                                           perturbed_likelihoods = perturbed_likelihoods,
                                                                          )

                variance_score           = np.var(perturbed_likelihoods) if (len(perturbed_likelihoods) > 1) else 0.0
                avg_perturbed_likelihood = np.mean(perturbed_likelihoods)
                
                logger.info(f"Stability: {stability_score:.3f}, Curvature: {curvature_score:.3f}")
            
            else:
                # Use meaningful defaults when perturbations fail
                stability_score          = 0.3  # Assume more human-like when no perturbations work
                curvature_score          = 0.3
                variance_score           = 0.05
                avg_perturbed_likelihood = original_likelihood * 0.9  # Assume some drop
                logger.warning("No valid perturbations, using fallback values")
            
            # Calculate likelihood ratio
            likelihood_ratio             = original_likelihood / avg_perturbed_likelihood if avg_perturbed_likelihood > 0 else 1.0
            
            # Chunk-based analysis for whole-text understanding
            chunk_stabilities            = self._calculate_chunk_stability(text       = processed_text, 
                                                                           chunk_size = 150,
                                                                          )

            stability_variance           = np.var(chunk_stabilities) if chunk_stabilities else 0.1 
            avg_chunk_stability          = np.mean(chunk_stabilities) if chunk_stabilities else stability_score
            
            # Better normalization to prevent extreme values
            normalized_stability         = min(1.0, max(0.0, stability_score))
            normalized_curvature         = min(1.0, max(0.0, curvature_score))
            normalized_likelihood_ratio  = min(3.0, max(0.33, likelihood_ratio)) / 3.0
            
            return {"original_likelihood"         : round(original_likelihood, 4),
                    "avg_perturbed_likelihood"    : round(avg_perturbed_likelihood, 4),
                    "likelihood_ratio"            : round(likelihood_ratio, 4),
                    "normalized_likelihood_ratio" : round(normalized_likelihood_ratio, 4),
                    "stability_score"             : round(normalized_stability, 4),
                    "curvature_score"             : round(normalized_curvature, 4),
                    "perturbation_variance"       : round(variance_score, 4),
                    "avg_chunk_stability"         : round(avg_chunk_stability, 4),
                    "stability_variance"          : round(stability_variance, 4),
                    "num_perturbations"           : len(perturbations),
                    "num_valid_perturbations"     : len(perturbed_likelihoods),
                    "num_chunks_analyzed"         : len(chunk_stabilities),
                   }
            
        except Exception as e:
            logger.warning(f"MultiPerturbationStability feature calculation failed: {repr(e)}")
            return self._get_default_features()
    

    def _calculate_likelihood(self, text: str) -> float:
        """
        Calculate proper log-likelihood using token probabilities
        Inspired by DetectGPT's likelihood calculation approach
        """
        try:
            # Check text length before tokenization
            if (len(text.strip()) < 10):
                return 2.0  # Return reasonable baseline

            if not self.gpt_model or not self.gpt_tokenizer:
                logger.warning("GPT model not available for likelihood calculation")
                return 2.0

            # Ensure tokenizer has pad token
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            # Tokenize text with proper settings
            encodings      = self.gpt_tokenizer(text, 
                                                return_tensors        = 'pt', 
                                                truncation            = True,
                                                max_length            = 256,
                                                padding               = True,
                                                return_attention_mask = True,
                                               )

            input_ids      = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            # Minimum tokens for meaningful analysis
            if ((input_ids.numel() == 0) or (input_ids.size(1) < 3)):
                return 2.0
            
            # Calculate proper log-likelihood using token probabilities
            with torch.no_grad():
                outputs        = self.gpt_model(input_ids, 
                                                attention_mask = attention_mask,
                                               )
                
                logits         = outputs.logits
                
                # Calculate log probabilities for each token
                log_probs      = torch.nn.functional.log_softmax(logits, dim = -1)
                
                # Get the log probability of each actual token
                log_likelihood = 0.0
                token_count    = 0
                
                for i in range(input_ids.size(1) - 1):
                    # Only consider non-padding tokens
                    if (attention_mask[0, i] == 1):       
                        token_id        = input_ids[0, i + 1]  # Next token prediction
                        log_prob        = log_probs[0, i, token_id]
                        log_likelihood += log_prob.item()
                        token_count    += 1
                
                # Normalize by token count to get average log likelihood per token
                if (token_count > 0):
                    avg_log_likelihood = log_likelihood / token_count

                else:
                    avg_log_likelihood = 0.0
            
            # Convert to positive scale and normalize
            # Typical GPT-2 log probabilities range from ~-10 to ~-2
            # Higher normalized value = more likely text
            normalized_likelihood = max(0.5, min(10.0, -avg_log_likelihood))
            
            return normalized_likelihood
            
        except Exception as e:
            logger.warning(f"Likelihood calculation failed: {repr(e)}")
            return 2.0  # Return reasonable baseline on error
    

    def _generate_perturbations(self, text: str, num_perturbations: int = 5) -> List[str]:
        """
        Generate perturbed versions of the text using multiple techniques:
        1. Word deletion (simple but effective)
        2. Word swapping (preserve meaning)
        3. DistilRoBERTa masked prediction (DetectGPT-inspired, using lighter model than T5)
        4. Synonym replacement (fallback)
        """
        perturbations = list()
        
        try:
            # Pre-process text for perturbation
            processed_text = self._preprocess_text_for_perturbation(text)
            words          = processed_text.split()
            
            if (len(words) < 3):
                return [processed_text]

            # Method 1: Simple word deletion (most reliable)
            if (len(words) > 5):
                for _ in range(min(3, num_perturbations)):
                    try:
                        # Delete random words (10-20% of text)
                        delete_count    = max(1, len(words) // 10)
                        indices_to_keep = np.random.choice(len(words), len(words) - delete_count, replace = False)
                        
                        perturbed_words = [words[i] for i in sorted(indices_to_keep)]
                        perturbed_text  = ' '.join(perturbed_words)
                        
                        if (self._is_valid_perturbation(perturbed_text, processed_text)):
                            perturbations.append(perturbed_text)
                            
                    except Exception as e:
                        logger.debug(f"Word deletion perturbation failed: {e}")
                        continue
            
            # Method 2: Word swapping
            if (len(words) > 4) and (len(perturbations) < num_perturbations):
                for _ in range(min(2, num_perturbations - len(perturbations))):
                    try:
                        perturbed_words = words.copy()
                        
                        # Swap random adjacent words
                        if (len(perturbed_words) >= 3):
                            swap_pos                                                 = np.random.randint(0, len(perturbed_words) - 2)
                            perturbed_words[swap_pos], perturbed_words[swap_pos + 1] = perturbed_words[swap_pos + 1], perturbed_words[swap_pos]
                        
                        perturbed_text = ' '.join(perturbed_words)
                        
                        if (self._is_valid_perturbation(perturbed_text, processed_text)):
                            perturbations.append(perturbed_text)
                            
                    except Exception as e:
                        logger.debug(f"Word swapping perturbation failed: {e}")
                        continue
            
            # Method 3: DistilRoBERTa-based masked word replacement (DetectGPT-inspired)
            if (self.mask_model and self.mask_tokenizer and (len(words) > 4) and len(perturbations) < num_perturbations):
                
                try:
                    roberta_perturbations = self._generate_roberta_masked_perturbations(text              = processed_text, 
                                                                                        words             = words, 
                                                                                        max_perturbations = num_perturbations - len(perturbations),
                                                                                       )
                    perturbations.extend(roberta_perturbations)
                    
                except Exception as e:
                    logger.warning(f"DistilRoBERTa masked perturbation failed: {repr(e)}")
            
            # Method 4: Synonym replacement as fallback
            if (len(perturbations) < num_perturbations):
                try:
                    synonym_perturbations = self._generate_synonym_perturbations(text              = processed_text, 
                                                                                 words             = words, 
                                                                                 max_perturbations = num_perturbations - len(perturbations),
                                                                                )
                    perturbations.extend(synonym_perturbations)
                    
                except Exception as e:
                    logger.debug(f"Synonym replacement failed: {repr(e)}")
            
            # Ensure we have at least some perturbations
            if not perturbations:
                # Fallback: create simple variations
                fallback_perturbations = self._generate_fallback_perturbations(text  = processed_text, 
                                                                               words = words,
                                                                              )
                perturbations.extend(fallback_perturbations)
            
            # Remove duplicates and ensure we don't exceed requested number
            unique_perturbations = list()
            
            for p in perturbations:
                if (p and (p != processed_text) and (p not in unique_perturbations) and (self._is_valid_perturbation(p, processed_text))):
                    unique_perturbations.append(p)
            
            return unique_perturbations[:num_perturbations]
            
        except Exception as e:
            logger.warning(f"Perturbation generation failed: {repr(e)}")
            return [text]  # Return at least the original text as fallback
    

    def _generate_roberta_masked_perturbations(self, text: str, words: List[str], max_perturbations: int) -> List[str]:
        """
        Generate perturbations using DistilRoBERTa mask filling
        This is inspired by DetectGPT but uses a lighter model (DistilRoBERTa instead of T5)
        """
        perturbations = list()
        
        try:
            # Use the proper DistilRoBERTa mask token from tokenizer
            if hasattr(self.mask_tokenizer, 'mask_token') and self.mask_tokenizer.mask_token:
                roberta_mask_token = self.mask_tokenizer.mask_token
            
            else:
                roberta_mask_token = "<mask>"  # Fallback
            
            # Select words to mask (avoid very short words and punctuation)
            candidate_positions = [i for i, word in enumerate(words) if (len(word) > 3) and word.isalpha() and word.lower() not in ['the', 'and', 'but', 'for', 'with']]
            
            if not candidate_positions:
                candidate_positions = [i for i, word in enumerate(words) if len(word) > 2]
            
            if not candidate_positions:
                return perturbations
            
            # Try multiple mask positions
            attempts          = min(max_perturbations * 2, len(candidate_positions))
            positions_to_try  = np.random.choice(candidate_positions, min(attempts, len(candidate_positions)), replace = False)
            
            for pos in positions_to_try:
                if (len(perturbations) >= max_perturbations):
                    break
                    
                try:
                    # Create masked text
                    masked_words      = words.copy()
                    original_word     = masked_words[pos]
                    masked_words[pos] = roberta_mask_token
                    masked_text       = ' '.join(masked_words)
                    
                    # DistilRoBERTa works better with proper sentence structure
                    if not masked_text.endswith(('.', '!', '?')):
                        masked_text += '.'
                    
                    # Tokenize with DistilRoBERTa-specific settings
                    inputs = self.mask_tokenizer(masked_text,
                                                 return_tensors = "pt",
                                                 truncation     = True,
                                                 max_length     = min(128, self.mask_tokenizer.model_max_length),
                                                 padding        = True,
                                                )
                    
                    # Move to appropriate device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get model predictions
                    with torch.no_grad():
                        outputs     = self.mask_model(**inputs)
                        predictions = outputs.logits
                    
                    # Get the mask token position
                    mask_token_index = torch.where(inputs["input_ids"][0] == self.mask_tokenizer.mask_token_id)[0]
                    
                    if (len(mask_token_index) == 0):
                        continue
                        
                    mask_token_index = mask_token_index[0]
                    
                    # Get top prediction
                    probs            = torch.nn.functional.softmax(predictions[0, mask_token_index], dim = -1)
                    top_tokens       = torch.topk(probs, 3, dim = -1)
                    
                    for token_id in top_tokens.indices:
                        predicted_token = self.mask_tokenizer.decode(token_id).strip()
                        
                        # Clean the predicted token
                        predicted_token = self._clean_roberta_token(predicted_token)
                        
                        if (predicted_token and (predicted_token != original_word) and (len(predicted_token) > 1)):
                            
                            # Replace the masked word
                            new_words      = words.copy()
                            new_words[pos] = predicted_token
                            new_text       = ' '.join(new_words)
                            
                            if (self._is_valid_perturbation(new_text, text)):
                                perturbations.append(new_text)
                                break  # Use first valid prediction
                    
                except Exception as e:
                    logger.debug(f"DistilRoBERTa mask filling failed for position {pos}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"DistilRoBERTa masked perturbations failed: {e}")
        
        return perturbations
    

    def _generate_synonym_perturbations(self, text: str, words: List[str], max_perturbations: int) -> List[str]:
        """
        Simple synonym replacement as fallback
        """
        perturbations = list()
        
        try:
            # Simple manual synonym dictionary for common words
            synonym_dict          = {'good'  : ['great', 'excellent', 'fine', 'nice'],
                                     'bad'   : ['poor', 'terrible', 'awful', 'horrible'],
                                     'big'   : ['large', 'huge', 'enormous', 'massive'],
                                     'small' : ['tiny', 'little', 'miniature', 'compact'],
                                     'fast'  : ['quick', 'rapid', 'speedy', 'brisk'],
                                     'slow'  : ['sluggish', 'leisurely', 'gradual', 'unhurried'],
                                    }
            
            # Find replaceable words
            replaceable_positions = [i for i, word in enumerate(words)  if word.lower() in synonym_dict]
            
            if not replaceable_positions:
                return perturbations
            
            positions_to_try      = np.random.choice(replaceable_positions, min(max_perturbations, len(replaceable_positions)), replace = False)
            
            for pos in positions_to_try:
                original_word = words[pos].lower()
                synonyms      = synonym_dict.get(original_word, [])
                
                if synonyms:
                    synonym        = np.random.choice(synonyms)
                    new_words      = words.copy()
                    new_words[pos] = synonym
                    new_text       = ' '.join(new_words)
                    
                    if (self._is_valid_perturbation(new_text, text)):
                        perturbations.append(new_text)
                        
        except Exception as e:
            logger.debug(f"Synonym replacement failed: {repr(e)}")
        
        return perturbations
    

    def _generate_fallback_perturbations(self, text: str, words: List[str]) -> List[str]:
        """
        Generate fallback perturbations when other methods fail
        """
        perturbations = list()
        
        try:
            # Remove first and last word
            if (len(words) > 3):
                perturbations.append(' '.join(words[1:-1]))
            
            # Remove first word only
            elif (len(words) > 1):
                perturbations.append(' '.join(words[1:]))
            
            # Capitalize/lowercase variations
            if text:
                perturbations.append(text.lower())
                perturbations.append(text.capitalize())
                
        except Exception as e:
            logger.debug(f"Fallback perturbation failed: {repr(e)}")
        
        return [p for p in perturbations if p and p != text][:3]
    

    def _calculate_stability_score(self, original_likelihood: float, perturbed_likelihoods: List[float]) -> float:
        """
        Calculate text stability score with improved normalization : AI text typically shows higher stability (larger drops) than human text
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= 0)):
            # Assume more human-like when no data
            return 0.3  
        
        # Calculate relative likelihood drops
        relative_drops = list()
        
        for pl in perturbed_likelihoods:
            if (pl > 0):
                # Use relative drop to handle scale differences
                relative_drop = (original_likelihood - pl) / original_likelihood
                
                # Clamp to [0, 1]
                relative_drops.append(max(0.0, min(1.0, relative_drop))) 
        
        if not relative_drops:
            return 0.3
        
        avg_relative_drop = np.mean(relative_drops)
        
        # Normalization based on empirical observations : AI text typically shows 20-60% drops, human text shows 10-30% drops
        if (avg_relative_drop > 0.5):
            # Strong AI indicator
            stability_score = 0.9  

        elif (avg_relative_drop > 0.3):
            # 0.6 to 0.9
            stability_score = 0.6 + (avg_relative_drop - 0.3) * 1.5  

        elif (avg_relative_drop > 0.15):
            # 0.3 to 0.6
            stability_score = 0.3 + (avg_relative_drop - 0.15) * 2.0  
        
        else:
            # 0.0 to 0.3
            stability_score = avg_relative_drop * 2.0  

        return min(1.0, max(0.0, stability_score))
    

    def _calculate_curvature_score(self, original_likelihood: float, perturbed_likelihoods: List[float]) -> float:
        """
        Calculate likelihood curvature score with better scaling : Measures how "curved" the likelihood surface is around the text
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= 0)):
            return 0.3
        
        # Calculate variance of likelihood changes
        likelihood_changes = [abs(original_likelihood - pl) for pl in perturbed_likelihoods]
        
        if (len(likelihood_changes) < 2):
            return 0.3
            
        change_variance = np.var(likelihood_changes)
        
        # Typical variance for meaningful analysis is around 0.1-0.5 : Adjusted scaling
        curvature_score = min(1.0, change_variance * 3.0)  
        
        return curvature_score
    

    def _calculate_chunk_stability(self, text: str, chunk_size: int = 150) -> List[float]:
        """
        Calculate stability across text chunks for whole-text analysis
        """
        stabilities = list()
        words       = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if (len(chunk) > 50):
                try:
                    chunk_likelihood = self._calculate_likelihood(text = chunk)
                    
                    if (chunk_likelihood > 0):
                        # Generate a simple perturbation for this chunk
                        chunk_words = chunk.split()
                        
                        if (len(chunk_words) > 5):
                            # Delete 10% of words
                            delete_count         = max(1, len(chunk_words) // 10)
                            indices_to_keep      = np.random.choice(len(chunk_words), len(chunk_words) - delete_count, replace=False)
                            perturbed_chunk      = ' '.join([chunk_words[i] for i in sorted(indices_to_keep)])
                            
                            perturbed_likelihood = self._calculate_likelihood(text = perturbed_chunk)

                            if (perturbed_likelihood > 0):
                                stability = (chunk_likelihood - perturbed_likelihood) / chunk_likelihood
                                stabilities.append(min(1.0, max(0.0, stability)))

                except Exception:
                    continue
        
        return stabilities
    

    def _analyze_stability_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze MultiPerturbationStability patterns with better feature weighting
        """
        # Check feature validity first
        required_features = ['stability_score', 'curvature_score', 'normalized_likelihood_ratio', 'stability_variance', 'perturbation_variance']
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > 0]
        
        if (len(valid_features) < 3):
            # Low confidence if insufficient features
            return 0.5, 0.3  


        # Initialize ai_indicator list
        ai_indicators    = list()
        
        # Better weighting based on feature reliability
        stability_weight = 0.3
        curvature_weight = 0.25
        ratio_weight     = 0.25
        variance_weight  = 0.2
        
        # High stability score suggests AI (larger likelihood drops)
        stability = features['stability_score']
        if (stability > 0.7):
            ai_indicators.append(0.9 * stability_weight)
        
        elif (stability > 0.5):
            ai_indicators.append(0.7 * stability_weight)
        
        elif (stability > 0.3):
            ai_indicators.append(0.5 * stability_weight)
        
        else:
            ai_indicators.append(0.2 * stability_weight)
        
        # High curvature score suggests AI
        curvature = features['curvature_score']
        if (curvature > 0.7):
            ai_indicators.append(0.8 * curvature_weight)
        
        elif (curvature > 0.5):
            ai_indicators.append(0.6 * curvature_weight)
        
        elif (curvature > 0.3):
            ai_indicators.append(0.4 * curvature_weight)
        
        else:
            ai_indicators.append(0.2 * curvature_weight)
        
        # High likelihood ratio suggests AI (original much more likely than perturbations)
        ratio = features['normalized_likelihood_ratio']
        if (ratio > 0.8):
            ai_indicators.append(0.9 * ratio_weight)
        
        elif (ratio > 0.6):
            ai_indicators.append(0.7 * ratio_weight)
        
        elif (ratio > 0.4):
            ai_indicators.append(0.5 * ratio_weight)
        
        else:
            ai_indicators.append(0.3 * ratio_weight)
        
        # Low stability variance suggests AI (consistent across chunks)
        stability_var = features['stability_variance']
        if (stability_var < 0.05):
            ai_indicators.append(0.8 * variance_weight)
        
        elif (stability_var < 0.1):
            ai_indicators.append(0.5 * variance_weight)

        else:
            ai_indicators.append(0.2 * variance_weight)
        
        # Calculate raw score and confidence
        if ai_indicators:
            raw_score  = sum(ai_indicators)
            confidence = 0.5 + (0.5 * (1.0 - (np.std([x / (weights := [stability_weight, curvature_weight, ratio_weight, variance_weight])[i] for i, x in enumerate(ai_indicators)]) if len(ai_indicators) > 1 else 0.5)))
       
        else:
            raw_score  = 0.5
            confidence = 0.3
            
        confidence = max(0.1, min(0.9, confidence))
        
        return raw_score, confidence
    

    def _calculate_mixed_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of mixed AI/Human content
        """
        mixed_indicators = list()
        
        # Moderate stability values might indicate mixing
        if (0.35 <= features['stability_score'] <= 0.55):
            mixed_indicators.append(0.3)
       
        else:
            mixed_indicators.append(0.0)
        
        # High stability variance suggests mixed content
        if (features['stability_variance'] > 0.15):
            mixed_indicators.append(0.4)

        elif (features['stability_variance'] > 0.1):
            mixed_indicators.append(0.2)

        else:
            mixed_indicators.append(0.0)
        
        # Inconsistent likelihood ratios
        if (0.5 <= features['normalized_likelihood_ratio'] <= 0.8):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        return min(0.3, np.mean(mixed_indicators)) if mixed_indicators else 0.0
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return more meaningful default features
        """
        return {"original_likelihood"         : 2.0,
                "avg_perturbed_likelihood"    : 1.8,
                "likelihood_ratio"            : 1.1,
                "normalized_likelihood_ratio" : 0.55,
                "stability_score"             : 0.3, 
                "curvature_score"             : 0.3,
                "perturbation_variance"       : 0.05,
                "avg_chunk_stability"         : 0.3,
                "stability_variance"          : 0.1,
                "num_perturbations"           : 0,
                "num_valid_perturbations"     : 0,
                "num_chunks_analyzed"         : 0,
               }
    

    def _preprocess_text_for_analysis(self, text: str) -> str:
        """
        Preprocess text for MultiPerturbationStability analysis
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Truncate very long texts
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    

    def _preprocess_text_for_perturbation(self, text: str) -> str:
        """
        Preprocess text specifically for perturbation generation
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # DistilRoBERTa works better with proper punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Truncate to safe length
        if (len(text) > 1000):
            sentences = text.split('. ')
            if (len(sentences) > 1):
                # Keep first few sentences
                text = '. '.join(sentences[:3]) + '.'
            
            else:
                text = text[:1000]
        
        return text
    

    def _clean_roberta_token(self, token: str) -> str:
        """
        Clean tokens from DistilRoBERTa tokenizer
        """
        if not token:
            return ""
        
        # Remove DistilRoBERTa-specific artifacts
        token = token.replace('Ġ', ' ')  # RoBERTa space marker
        token = token.replace('</s>', '')
        token = token.replace('<s>', '')
        token = token.replace('<pad>', '')
        token = token.replace('<mask>', '')
        
        # Remove leading/trailing whitespace
        token = token.strip()
        
        # Only remove punctuation if token is ONLY punctuation
        if token and not token.replace('.', '').replace(',', '').replace('!', '').replace('?', '').strip():
            return ""
        
        # Keep the token if it has at least 2 alphanumeric characters
        if sum(c.isalnum() for c in token) >= 2:
            return token
        
        return ""
    

    def _is_valid_perturbation(self, perturbed_text: str, original_text: str) -> bool:
        """
        Check if a perturbation is valid (more lenient validation)
        """
        if (not perturbed_text or not perturbed_text.strip()):
            return False
        
        # Must be different from original
        if (perturbed_text == original_text):
            return False
        
        # Lenient length check
        if (len(perturbed_text) < len(original_text) * 0.3):
            return False
        
        # Must have some actual content
        if len(perturbed_text.strip()) < 5:
            return False
        
        return True
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.gpt_model      = None
        self.gpt_tokenizer  = None
        self.mask_model     = None
        self.mask_tokenizer = None

        super().cleanup()


# Export
__all__ = ["MultiPerturbationStabilityMetric"]