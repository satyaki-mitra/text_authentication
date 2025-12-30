# DEPENDENCIES
import torch
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from config.enums import Domain
from config.schemas import MetricResult
from metrics.base_metric import BaseMetric
from models.model_manager import get_model_manager
from config.threshold_config import get_threshold_for_domain
from config.constants import multi_perturbation_stability_metric_params


class MultiPerturbationStabilityMetric(BaseMetric):
    """
    Multi-Perturbation Stability Metric (MPSM) 
    
    A hybrid approach for combining multiple perturbation techniques for robust synthetic-generated text detection
    
    Measures:
    - Text stability under random perturbations
    - Likelihood curvature analysis
    - Masked token prediction analysis

    Perturbation Methods:
    - Word deletion & swapping
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
        self.params         = multi_perturbation_stability_metric_params
    

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
            if ((not text) or (len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_ANALYSIS)):
                return self._default_result(error = "Text too short for MultiPerturbationStability analysis")
            
            # Get domain-specific thresholds
            domain                                  = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                       = get_threshold_for_domain(domain)
            multi_perturbation_stability_thresholds = domain_thresholds.multi_perturbation_stability
            
            # Check if we should run this computationally expensive metric
            if (kwargs.get('skip_expensive', False)):
                logger.info("Skipping MultiPerturbationStability due to computational constraints")
                return self._default_result(error = "Skipped for performance")
            
            # Calculate MultiPerturbationStability features
            features                                    = self._calculate_stability_features(text = text)
            
            # Calculate raw MultiPerturbationStability score (0-1 scale)
            raw_stability_score, confidence             = self._analyze_stability_patterns(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob = self._apply_domain_thresholds(raw_score  = raw_stability_score, 
                                                                                        thresholds = multi_perturbation_stability_thresholds, 
                                                                                        features   = features,
                                                                                       )
            
            # Apply confidence multiplier from domain thresholds
            confidence                                 *= multi_perturbation_stability_thresholds.confidence_multiplier
            confidence                                  = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
            
            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'        : domain.value,
                                                         'synthetic_threshold': multi_perturbation_stability_thresholds.synthetic_threshold,
                                                         'authentic_threshold': multi_perturbation_stability_thresholds.authentic_threshold,
                                                         'raw_score'          : raw_stability_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in MultiPerturbationStability computation: {repr(e)}")
            return self._default_result(error = str(e))
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        synthetic_threshold = thresholds.synthetic_threshold
        authentic_threshold = thresholds.authentic_threshold
        
        # Calculate probabilities based on threshold distances
        if (raw_score >= synthetic_threshold):
            # Above synthetic threshold - strongly synthetic
            distance_from_threshold = raw_score - synthetic_threshold
            synthetic_prob          = self.params.STRONG_SYNTHETIC_BASE_PROB + (distance_from_threshold * self.params.WEAK_PROBABILITY_ADJUSTMENT)
            authentic_prob          = self.params.UNCERTAIN_AUTHENTIC_RANGE_START - (distance_from_threshold * self.params.WEAK_PROBABILITY_ADJUSTMENT)

        elif (raw_score <= authentic_threshold):
            # Below authentic threshold - strongly authentic
            distance_from_threshold = authentic_threshold - raw_score
            synthetic_prob          = self.params.UNCERTAIN_SYNTHETIC_RANGE_START - (distance_from_threshold * self.params.WEAK_PROBABILITY_ADJUSTMENT)
            authentic_prob          = self.params.STRONG_AUTHENTIC_BASE_PROB + (distance_from_threshold * self.params.WEAK_PROBABILITY_ADJUSTMENT)

        else:
            # Between thresholds - uncertain zone
            range_width = synthetic_threshold - authentic_threshold

            if (range_width > self.params.ZERO_TOLERANCE):
                position_in_range = (raw_score - authentic_threshold) / range_width
                synthetic_prob    = self.params.UNCERTAIN_SYNTHETIC_RANGE_START + (position_in_range * self.params.UNCERTAIN_RANGE_WIDTH)
                authentic_prob    = self.params.UNCERTAIN_AUTHENTIC_RANGE_START - (position_in_range * self.params.UNCERTAIN_RANGE_WIDTH)
            
            else:
                synthetic_prob = self.params.NEUTRAL_PROBABILITY
                authentic_prob = self.params.NEUTRAL_PROBABILITY
        
        # Ensure probabilities are valid
        synthetic_prob = max(self.params.MIN_PROBABILITY, min(self.params.MAX_PROBABILITY, synthetic_prob))
        authentic_prob = max(self.params.MIN_PROBABILITY, min(self.params.MAX_PROBABILITY, authentic_prob))
        
        # Calculate hybrid probability based on stability variance
        hybrid_prob    = self._calculate_hybrid_probability(features)
        
        # Normalize to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob

        if (total > self.params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob
    

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
            perturbations         = self._generate_perturbations(text = processed_text)
            logger.debug(f"Generated {len(perturbations)} perturbations")

            perturbed_likelihoods = list()
            
            for idx, perturbed_text in enumerate(perturbations):
                if (perturbed_text and (perturbed_text != processed_text)):
                    likelihood = self._calculate_likelihood(text = perturbed_text)
                    
                    if (likelihood > self.params.ZERO_TOLERANCE):
                        perturbed_likelihoods.append(likelihood)
                        logger.debug(f"Perturbation {idx}: likelihood={likelihood:.4f}")
            
            logger.info(f"Valid perturbations: {len(perturbed_likelihoods)}/{len(perturbations)}")
            
            # Calculate stability metrics
            if perturbed_likelihoods and (len(perturbed_likelihoods) >= self.params.MIN_VALID_PERTURBATIONS):
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
                stability_score          = self.params.DEFAULT_STABILITY_SCORE  # Assume more authentic-like when no perturbations work
                curvature_score          = self.params.DEFAULT_CURVATURE_SCORE
                variance_score           = self.params.DEFAULT_PERTURBATION_VARIANCE
                avg_perturbed_likelihood = original_likelihood * 0.9  # Assume some drop
                logger.warning("No valid perturbations, using fallback values")
            
            # Calculate likelihood ratio
            likelihood_ratio             = original_likelihood / avg_perturbed_likelihood if avg_perturbed_likelihood > self.params.ZERO_TOLERANCE else 1.0
            
            # Chunk-based analysis for whole-text understanding
            chunk_stabilities            = self._calculate_chunk_stability(text = processed_text)
            stability_variance           = np.var(chunk_stabilities) if chunk_stabilities else self.params.DEFAULT_STABILITY_VARIANCE 
            avg_chunk_stability          = np.mean(chunk_stabilities) if chunk_stabilities else stability_score
            
            # Better normalization to prevent extreme values
            normalized_stability         = min(1.0, max(0.0, stability_score))
            normalized_curvature         = min(1.0, max(0.0, curvature_score))
            normalized_likelihood_ratio  = min(self.params.MAX_LIKELIHOOD_RATIO, max(self.params.MIN_LIKELIHOOD_RATIO, likelihood_ratio)) / self.params.MAX_LIKELIHOOD_RATIO
            
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
            if (len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_PERTURBATION):
                # Return reasonable baseline
                return self.params.DEFAULT_LIKELIHOOD  

            if not self.gpt_model or not self.gpt_tokenizer:
                logger.warning("GPT model not available for likelihood calculation")
                return self.params.DEFAULT_LIKELIHOOD

            # Ensure tokenizer has pad token
            if self.gpt_tokenizer.pad_token is None:
                self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            
            # Tokenize text with proper settings
            encodings      = self.gpt_tokenizer(text, 
                                                return_tensors        = 'pt', 
                                                truncation            = True,
                                                max_length            = self.params.MAX_TOKEN_LENGTH,
                                                padding               = True,
                                                return_attention_mask = True,
                                               )

            input_ids      = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            # Minimum tokens for meaningful analysis
            if ((input_ids.numel() == 0) or (input_ids.size(1) < self.params.MIN_TOKENS_FOR_LIKELIHOOD)):
                return self.params.DEFAULT_LIKELIHOOD
            
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
            normalized_likelihood = max(self.params.MIN_LIKELIHOOD, min(self.params.MAX_LIKELIHOOD, -avg_log_likelihood))
            
            return normalized_likelihood
            
        except Exception as e:
            logger.warning(f"Likelihood calculation failed: {repr(e)}")
            # Return reasonable baseline on error
            return self.params.DEFAULT_LIKELIHOOD  
    

    def _generate_perturbations(self, text: str) -> List[str]:
        """
        Generate perturbed versions of the text using multiple techniques:
        1. Word deletion (simple but effective)
        2. Word swapping (preserve meaning)
        3. DistilRoBERTa masked prediction (DetectGPT-inspired, using lighter model than T5)
        4. Synonym replacement (fallback)
        """
        perturbations = list()
        num_perturbations = self.params.NUM_PERTURBATIONS
        
        try:
            # Pre-process text for perturbation
            processed_text = self._preprocess_text_for_perturbation(text)
            words          = processed_text.split()
            
            if (len(words) < self.params.MIN_WORDS_FOR_PERTURBATION):
                return [processed_text]

            # Method 1: Simple word deletion (most reliable)
            if (len(words) > self.params.MIN_WORDS_FOR_DELETION):
                for _ in range(min(3, num_perturbations)):
                    try:
                        # Delete random words
                        delete_count    = max(1, int(len(words) * self.params.PERTURBATION_DELETION_RATIO))
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
                    roberta_perturbations = self._generate_roberta_masked_perturbations(text  = processed_text, 
                                                                                        words = words,
                                                                                       )
                    perturbations.extend(roberta_perturbations)
                    
                except Exception as e:
                    logger.warning(f"DistilRoBERTa masked perturbation failed: {repr(e)}")
            
            # Method 4: Synonym replacement as fallback
            if (len(perturbations) < num_perturbations):
                try:
                    synonym_perturbations = self._generate_synonym_perturbations(text = processed_text, words = words)
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
    

    def _generate_roberta_masked_perturbations(self, text: str, words: List[str]) -> List[str]:
        """
        Generate perturbations using DistilRoBERTa mask filling
        - This is inspired by DetectGPT but uses a lighter model (DistilRoBERTa instead of T5)
        """
        perturbations     = list()
        max_perturbations = min(self.params.MAX_PERTURBATION_ATTEMPTS, self.params.NUM_PERTURBATIONS - len(perturbations))
        
        try:
            # Use the proper DistilRoBERTa mask token from tokenizer
            if hasattr(self.mask_tokenizer, 'mask_token') and self.mask_tokenizer.mask_token:
                roberta_mask_token = self.mask_tokenizer.mask_token
            
            else:
                # Fallback
                roberta_mask_token = "<mask>"  
            
            # Select words to mask (avoid very short words and punctuation)
            candidate_positions = [i for i, word in enumerate(words) if (len(word) > 3) and word.isalpha() and word.lower() not in self.params.COMMON_WORDS_TO_AVOID]
            
            if not candidate_positions:
                candidate_positions = [i for i, word in enumerate(words) if (len(word) > 2)]
            
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
                    inputs            = self.mask_tokenizer(masked_text,
                                                            return_tensors = "pt",
                                                            truncation     = True,
                                                            max_length     = min(self.params.MAX_ROBERTA_TOKEN_LENGTH, self.mask_tokenizer.model_max_length),
                                                            padding        = True,
                                                           )
                    
                    # Move to appropriate device
                    inputs            = {k: v.to(self.device) for k, v in inputs.items()}
                    
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
                    top_tokens       = torch.topk(probs, self.params.ROBBERTA_TOP_K_PREDICTIONS, dim = -1)
                    
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
    

    def _generate_synonym_perturbations(self, text: str, words: List[str]) -> List[str]:
        """
        Simple synonym replacement as fallback
        """
        perturbations     = list()
        max_perturbations = self.params.NUM_PERTURBATIONS - len(perturbations)
        
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
            
            positions_to_try      = np.random.choice(replaceable_positions, 
                                                     min(max_perturbations, len(replaceable_positions)), 
                                                     replace = False,
                                                    )
            
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
        Calculate text stability score with normalization : synthetic text typically shows larger likelihood drops under perturbation than authentic text
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= self.params.ZERO_TOLERANCE)):
            # Assume more authentic-like when no data
            return self.params.DEFAULT_STABILITY_SCORE  
        
        # Calculate relative likelihood drops
        relative_drops = list()
        
        for pl in perturbed_likelihoods:
            if (pl > self.params.ZERO_TOLERANCE):
                # Use relative drop to handle scale differences
                relative_drop = (original_likelihood - pl) / original_likelihood
                
                # Clamp to [0, 1]
                relative_drops.append(max(0.0, min(1.0, relative_drop))) 
        
        if not relative_drops:
            return self.params.DEFAULT_STABILITY_SCORE
        
        avg_relative_drop = np.mean(relative_drops)
        
        # Normalization based on empirical observations : synthetic text typically shows larger drops
        if (avg_relative_drop > self.params.RELATIVE_DROP_HIGH_THRESHOLD):
            # Strong synthetic indicator
            stability_score = self.params.STABILITY_HIGH_THRESHOLD  

        elif (avg_relative_drop > self.params.RELATIVE_DROP_MEDIUM_THRESHOLD):
            # Intermediate values
            stability_score = self.params.STABILITY_MEDIUM_THRESHOLD + (avg_relative_drop - self.params.RELATIVE_DROP_MEDIUM_THRESHOLD) * 1.5  

        elif (avg_relative_drop > self.params.RELATIVE_DROP_LOW_THRESHOLD):
            # Lower values
            stability_score = self.params.STABILITY_LOW_THRESHOLD + (avg_relative_drop - self.params.RELATIVE_DROP_LOW_THRESHOLD) * 2.0  
        
        else:
            # Very low values
            stability_score = avg_relative_drop * 2.0  

        return min(1.0, max(0.0, stability_score))
    

    def _calculate_curvature_score(self, original_likelihood: float, perturbed_likelihoods: List[float]) -> float:
        """
        Calculate likelihood curvature score with better scaling : Measures how "curved" the likelihood surface is around the text
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= self.params.ZERO_TOLERANCE)):
            return self.params.DEFAULT_CURVATURE_SCORE
        
        # Calculate variance of likelihood changes
        likelihood_changes = [abs(original_likelihood - pl) for pl in perturbed_likelihoods]
        
        if (len(likelihood_changes) < 2):
            return self.params.DEFAULT_CURVATURE_SCORE
            
        change_variance = np.var(likelihood_changes)
        
        # Typical variance for meaningful analysis
        curvature_score = min(1.0, change_variance * self.params.CURVATURE_SCALING_FACTOR)  
        
        return curvature_score
    

    def _calculate_chunk_stability(self, text: str) -> List[float]:
        """
        Calculate stability across text chunks for whole-text analysis
        """
        stabilities = list()
        words       = text.split()
        chunk_size = self.params.CHUNK_SIZE_WORDS
        overlap = int(chunk_size * self.params.CHUNK_OVERLAP_RATIO)
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if (len(chunk) > self.params.MIN_CHUNK_LENGTH):
                try:
                    chunk_likelihood = self._calculate_likelihood(text = chunk)
                    
                    if (chunk_likelihood > self.params.ZERO_TOLERANCE):
                        # Generate a simple perturbation for this chunk
                        chunk_words = chunk.split()
                        
                        if (len(chunk_words) > self.params.MIN_WORDS_FOR_DELETION):
                            # Delete a percentage of words
                            delete_count         = max(1, int(len(chunk_words) * self.params.CHUNK_DELETION_RATIO))
                            indices_to_keep      = np.random.choice(len(chunk_words), len(chunk_words) - delete_count, replace=False)
                            perturbed_chunk      = ' '.join([chunk_words[i] for i in sorted(indices_to_keep)])
                            
                            perturbed_likelihood = self._calculate_likelihood(text = perturbed_chunk)

                            if (perturbed_likelihood > self.params.ZERO_TOLERANCE):
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
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > self.params.ZERO_TOLERANCE]
        
        if (len(valid_features) < self.params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return self.params.NEUTRAL_PROBABILITY, self.params.LOW_FEATURE_CONFIDENCE


        # Initialize synthetic_indicator list
        synthetic_indicators    = list()
        
        # Better weighting based on feature reliability
        stability = features['stability_score']
        if (stability > self.params.STABILITY_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.STABILITY_STRONG_THRESHOLD * self.params.STABILITY_WEIGHT)
        
        elif (stability > self.params.STABILITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.STABILITY_MEDIUM_STRONG_THRESHOLD * self.params.STABILITY_WEIGHT)
        
        elif (stability > self.params.STABILITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.STABILITY_MODERATE_THRESHOLD * self.params.STABILITY_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.STABILITY_WEAK_THRESHOLD * self.params.STABILITY_WEIGHT)
        
        # High curvature score suggests synthetic
        curvature = features['curvature_score']
        if (curvature > self.params.CURVATURE_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.CURVATURE_STRONG_THRESHOLD * self.params.CURVATURE_WEIGHT)
        
        elif (curvature > self.params.CURVATURE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.CURVATURE_MEDIUM_THRESHOLD * self.params.CURVATURE_WEIGHT)
        
        elif (curvature > self.params.CURVATURE_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.CURVATURE_MODERATE_THRESHOLD * self.params.CURVATURE_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.CURVATURE_WEAK_THRESHOLD * self.params.CURVATURE_WEIGHT)
        
        # High likelihood ratio suggests synthetic (original much more likely than perturbations)
        ratio = features['normalized_likelihood_ratio']
        if (ratio > self.params.LIKELIHOOD_RATIO_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.RATIO_STRONG_THRESHOLD * self.params.RATIO_WEIGHT)
        
        elif (ratio > self.params.LIKELIHOOD_RATIO_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.RATIO_MEDIUM_THRESHOLD * self.params.RATIO_WEIGHT)
        
        elif (ratio > self.params.LIKELIHOOD_RATIO_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.RATIO_MODERATE_THRESHOLD * self.params.RATIO_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.RATIO_WEAK_THRESHOLD * self.params.RATIO_WEIGHT)
        
        # Low stability variance suggests synthetic (consistent across chunks)
        stability_var = features['stability_variance']
        if (stability_var < self.params.STABILITY_VARIANCE_VERY_LOW):
            synthetic_indicators.append(self.params.VARIANCE_STRONG_THRESHOLD * self.params.VARIANCE_WEIGHT)
        
        elif (stability_var < self.params.STABILITY_VARIANCE_LOW):
            synthetic_indicators.append(self.params.VARIANCE_MODERATE_THRESHOLD * self.params.VARIANCE_WEIGHT)

        else:
            synthetic_indicators.append(self.params.VARIANCE_WEAK_THRESHOLD * self.params.VARIANCE_WEIGHT)
        
        # Calculate raw score and confidence
        if synthetic_indicators:
            total_weight = (self.params.STABILITY_WEIGHT + self.params.CURVATURE_WEIGHT + self.params.RATIO_WEIGHT + self.params.VARIANCE_WEIGHT)
            raw_score    = sum(synthetic_indicators) / total_weight
            weights      = [self.params.STABILITY_WEIGHT, self.params.CURVATURE_WEIGHT, self.params.RATIO_WEIGHT, self.params.VARIANCE_WEIGHT]
            confidence   = self.params.CONFIDENCE_BASE + (self.params.CONFIDENCE_STD_FACTOR * (1.0 - (np.std([x / weights[i] for i, x in enumerate(synthetic_indicators)]) if len(synthetic_indicators) > 1 else 0.5)))
       
        else:
            raw_score  = self.params.NEUTRAL_PROBABILITY
            confidence = self.params.LOW_FEATURE_CONFIDENCE
            
        confidence = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content
        """
        hybrid_indicators = list()
        
        # Moderate stability values might indicate mixing
        if (self.params.STABILITY_MIXED_MIN <= features['stability_score'] <= self.params.STABILITY_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)
       
        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # High stability variance suggests mixed content
        if (features['stability_variance'] > self.params.STABILITY_VARIANCE_MIXED_HIGH):
            hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)

        elif (features['stability_variance'] > self.params.STABILITY_VARIANCE_MIXED_MEDIUM):
            hybrid_indicators.append(self.params.VERY_WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # Inconsistent likelihood ratios
        if (self.params.LIKELIHOOD_RATIO_MIXED_MIN <= features['normalized_likelihood_ratio'] <= self.params.LIKELIHOOD_RATIO_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        hybrid_prob = np.mean(hybrid_indicators) if hybrid_indicators else 0.0
        return min(self.params.MAX_HYBRID_PROBABILITY, hybrid_prob)
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return more meaningful default features
        """
        return {"original_likelihood"         : self.params.DEFAULT_ORIGINAL_LIKELIHOOD,
                "avg_perturbed_likelihood"    : self.params.DEFAULT_AVG_PERTURBED_LIKELIHOOD,
                "likelihood_ratio"            : self.params.DEFAULT_LIKELIHOOD_RATIO,
                "normalized_likelihood_ratio" : self.params.DEFAULT_NORMALIZED_LIKELIHOOD_RATIO,
                "stability_score"             : self.params.DEFAULT_STABILITY_SCORE, 
                "curvature_score"             : self.params.DEFAULT_CURVATURE_SCORE,
                "perturbation_variance"       : self.params.DEFAULT_PERTURBATION_VARIANCE,
                "avg_chunk_stability"         : self.params.DEFAULT_AVG_CHUNK_STABILITY,
                "stability_variance"          : self.params.DEFAULT_STABILITY_VARIANCE,
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
        if len(text) > self.params.MAX_TEXT_LENGTH_FOR_ANALYSIS:
            text = text[:self.params.MAX_TEXT_LENGTH_FOR_ANALYSIS] + "..."
        
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
        if (len(text) > self.params.MAX_TEXT_LENGTH_FOR_PERTURBATION):
            sentences = text.split('. ')
            if (len(sentences) > 1):
                # Keep first few sentences
                text = '. '.join(sentences[:3]) + '.'
            
            else:
                text = text[:self.params.MAX_TEXT_LENGTH_FOR_PERTURBATION]
        
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
        if len(perturbed_text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_PERTURBATION:
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