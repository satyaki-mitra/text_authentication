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
    
    A hybrid approach combining multiple perturbation techniques for robust synthetic-generated text detection
    based on proper statistical foundations and DetectGPT methodology.
    
    Key Concept:
    - Synthetic text has smoother log-likelihood surfaces (more stable under perturbation)
    - Human text has rougher log-likelihood surfaces (less stable under perturbation)
    
    Measures:
    - Stability Score: Mean absolute log-probability difference between original and perturbed texts
      → Lower stability indicates synthetic text (text remains predictable after perturbations)
      → Higher stability indicates authentic text (text becomes less predictable after perturbations)
    
    - Curvature Score: Variance of log-probability differences across perturbations
      → Lower curvature indicates smoother likelihood surface (more synthetic)
      → Higher curvature indicates rougher likelihood surface (more authentic)
    
    - Variance Analysis: Consistency of stability across text chunks

    Perturbation Methods:
    - Word deletion & swapping
    - DistilRoBERTa mask filling (DetectGPT-inspired, lighter than T5)
    - Synonym replacement
    - Chunk-based stability analysis
    """
    def __init__(self):
        super().__init__(name        = "multi_perturbation_stability",
                         description = "Text stability analysis using log-probability perturbations",
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
            
            # Load GPT-2 model for log-probability calculation
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
                gpt_log_prob = self._calculate_log_probability(text = test_text)
                logger.info(f"GPT-2 test - Log-probability: {gpt_log_prob:.4f}")
            
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
        Compute MultiPerturbationStability analysis
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
        Calculate comprehensive MultiPerturbationStability features
        """
        if not self.gpt_model or not self.gpt_tokenizer:
            return self._get_default_features()
        
        try:
            # Preprocess text for better analysis
            processed_text        = self._preprocess_text_for_analysis(text = text)
            
            # Calculate original text log-probability
            original_log_prob     = self._calculate_log_probability(text = processed_text)
            logger.debug(f"Original log-probability: {original_log_prob:.4f}")
            
            # Generate perturbations and calculate perturbed log-probabilities
            perturbations         = self._generate_perturbations(text = processed_text)
            logger.debug(f"Generated {len(perturbations)} perturbations")

            perturbed_log_probs   = list()
            
            for idx, perturbed_text in enumerate(perturbations):
                if (perturbed_text and (perturbed_text != processed_text)):
                    log_prob = self._calculate_log_probability(text = perturbed_text)
                    
                    if (abs(log_prob) > self.params.ZERO_TOLERANCE):
                        perturbed_log_probs.append(log_prob)
                        logger.debug(f"Perturbation {idx}: log_prob={log_prob:.4f}")
            
            logger.info(f"Valid perturbations: {len(perturbed_log_probs)}/{len(perturbations)}")
            
            # Calculate stability metrics
            if perturbed_log_probs and (len(perturbed_log_probs) >= self.params.MIN_VALID_PERTURBATIONS):
                # STABILITY: Mean absolute log-probability difference
                stability_score          = self._calculate_stability_score(original_log_prob   = original_log_prob, 
                                                                           perturbed_log_probs = perturbed_log_probs,
                                                                          )

                # CURVATURE: Variance of log-probability differences
                curvature_score          = self._calculate_curvature_score(original_log_prob   = original_log_prob, 
                                                                           perturbed_log_probs = perturbed_log_probs,
                                                                          )

                variance_score           = np.var(perturbed_log_probs) if (len(perturbed_log_probs) > 1) else 0.0
                avg_perturbed_log_prob   = np.mean(perturbed_log_probs)
                
                logger.info(f"Stability: {stability_score:.3f}, Curvature: {curvature_score:.3f}")
            
            else:
                # Use meaningful defaults when perturbations fail
                stability_score          = self.params.DEFAULT_STABILITY_SCORE  # Assume neutral when no perturbations work
                curvature_score          = self.params.DEFAULT_CURVATURE_SCORE
                variance_score           = self.params.DEFAULT_PERTURBATION_VARIANCE
                avg_perturbed_log_prob   = original_log_prob * 1.1  # Assume slight increase
                logger.warning("No valid perturbations, using fallback values")
            
            # Chunk-based analysis for whole-text understanding
            chunk_stabilities            = self._calculate_chunk_stability(text = processed_text)
            stability_variance           = np.var(chunk_stabilities) if chunk_stabilities else self.params.DEFAULT_STABILITY_VARIANCE 
            avg_chunk_stability          = np.mean(chunk_stabilities) if chunk_stabilities else stability_score
            
            return {"original_log_prob"         : round(original_log_prob, 4),
                    "avg_perturbed_log_prob"    : round(avg_perturbed_log_prob, 4),
                    "stability_score"           : round(stability_score, 4),
                    "curvature_score"           : round(curvature_score, 4),
                    "perturbation_variance"     : round(variance_score, 4),
                    "avg_chunk_stability"       : round(avg_chunk_stability, 4),
                    "stability_variance"        : round(stability_variance, 4),
                    "num_perturbations"         : len(perturbations),
                    "num_valid_perturbations"   : len(perturbed_log_probs),
                    "num_chunks_analyzed"       : len(chunk_stabilities),
                   }
            
        except Exception as e:
            logger.warning(f"MultiPerturbationStability feature calculation failed: {repr(e)}")
            return self._get_default_features()
    

    def _calculate_log_probability(self, text: str) -> float:
        """
        Calculate average negative log-probability per token using cross-entropy loss

        Mathematical foundation:
        ------------------------
            negative_log_prob     = -log P(token | context) = CrossEntropyLoss
            avg_negative_log_prob = mean(negative_log_prob over all tokens)
        
        Returns: 
        --------
            float in range [~1.0, ~15.0] where:
            - Lower values (~1-3)   = very predictable text (potentially synthetic)
            - Higher values (~5-10) = less predictable text (potentially human)
            - Values are positive (absolute value of negative log-probability)
        """
        try:
            # Check text length before tokenization
            if (len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_PERTURBATION):
                # Return reasonable baseline
                return self.params.DEFAULT_LOG_PROB  

            if not self.gpt_model or not self.gpt_tokenizer:
                logger.warning("GPT model not available for log-probability calculation")
                return self.params.DEFAULT_LOG_PROB

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
                return self.params.DEFAULT_LOG_PROB
            
            # Calculate negative log-likelihood using PROPER cross-entropy loss
            with torch.no_grad():
                outputs = self.gpt_model(input_ids, 
                                        attention_mask = attention_mask,
                                       )
                
                logits  = outputs.logits
                
                # Shift for next-token prediction (standard language modeling)
                shift_logits    = logits[:, :-1, :].contiguous()
                shift_labels    = input_ids[:, 1:].contiguous()
                shift_attention = attention_mask[:, 1:].contiguous()
                
                # Calculate cross-entropy loss (negative log-likelihood) per token
                loss_fct        = torch.nn.CrossEntropyLoss(reduction = 'none')
                losses          = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                          shift_labels.view(-1),
                                         )
                
                # Reshape and mask out padding tokens
                losses          = losses.view(shift_labels.size())
                masked_losses   = losses * shift_attention
                
                # Average over non-padding tokens
                num_tokens      = shift_attention.sum()
                
                if (num_tokens > 0):
                    avg_neg_log_prob = masked_losses.sum() / num_tokens
                
                else:
                    avg_neg_log_prob = torch.tensor(self.params.DEFAULT_LOG_PROB)
                
                result = avg_neg_log_prob.item()
            
            # Sanity check: log-probabilities should be in reasonable range
            result = max(self.params.LOG_PROB_SANITY_MAX, 
                        min(abs(self.params.LOG_PROB_SANITY_MIN), abs(result)))
            
            # Return positive value for easier interpretation
            return abs(result)
            
        except Exception as e:
            logger.warning(f"Log-probability calculation failed: {repr(e)}")
            # Return reasonable baseline on error
            return self.params.DEFAULT_LOG_PROB  
    

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
    

    def _calculate_stability_score(self, original_log_prob: float, perturbed_log_probs: List[float]) -> float:
        """
        Stability calculation based on DetectGPT methodology
        
        Stability = mean(|original_log_prob - perturbed_log_prob|)
        
        Interpretation:
        - Low stability (< 0.5) = small differences = smooth surface = SYNTHETIC
        - High stability (> 1.5) = large differences = rough surface = AUTHENTIC
        """
        if not perturbed_log_probs:
            return self.params.DEFAULT_STABILITY_SCORE
        
        # Calculate absolute differences 
        differences = [abs(original_log_prob - plp) for plp in perturbed_log_probs]
        
        # Mean absolute difference (proper stability measure)
        mean_diff   = np.mean(differences)
        
        # Return raw mean difference (no arbitrary scaling)
        return mean_diff
    

    def _calculate_curvature_score(self, original_log_prob: float, perturbed_log_probs: List[float]) -> float:
        """
        Curvature calculation
        
        Curvature = variance(|original_log_prob - perturbed_log_prob|) * scaling_factor
        
        Measures the consistency of perturbation effects:
        - Low curvature (< 0.1) = consistent effects = smooth = SYNTHETIC
        - High curvature (> 0.5) = inconsistent effects = rough = AUTHENTIC
        """
        if (len(perturbed_log_probs) < 2):
            return self.params.DEFAULT_CURVATURE_SCORE
        
        # Calculate absolute differences
        differences     = [abs(original_log_prob - plp) for plp in perturbed_log_probs]
        
        # Variance of differences (measures surface roughness)
        variance        = np.var(differences)
        
        # Scale for interpretability (variance is typically very small)
        scaled_variance = variance * self.params.CURVATURE_SCALING_FACTOR
        
        return min(1.0, scaled_variance)
    

    def _calculate_chunk_stability(self, text: str) -> List[float]:
        """
        Calculate stability across text chunks for whole-text analysis
        """
        stabilities = list()
        words       = text.split()
        chunk_size  = self.params.CHUNK_SIZE_WORDS
        overlap     = int(chunk_size * self.params.CHUNK_OVERLAP_RATIO)
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if (len(chunk) > self.params.MIN_CHUNK_LENGTH):
                try:
                    chunk_log_prob = self._calculate_log_probability(text = chunk)
                    
                    if (abs(chunk_log_prob) > self.params.ZERO_TOLERANCE):
                        # Generate a simple perturbation for this chunk
                        chunk_words = chunk.split()
                        
                        if (len(chunk_words) > self.params.MIN_WORDS_FOR_DELETION):
                            # Delete a percentage of words
                            delete_count         = max(1, int(len(chunk_words) * self.params.CHUNK_DELETION_RATIO))
                            indices_to_keep      = np.random.choice(len(chunk_words), len(chunk_words) - delete_count, replace=False)
                            perturbed_chunk      = ' '.join([chunk_words[i] for i in sorted(indices_to_keep)])
                            
                            perturbed_log_prob   = self._calculate_log_probability(text = perturbed_chunk)

                            if (abs(perturbed_log_prob) > self.params.ZERO_TOLERANCE):
                                stability = abs(chunk_log_prob - perturbed_log_prob)
                                stabilities.append(stability)

                except Exception:
                    continue
        
        return stabilities
    

    def _analyze_stability_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze MultiPerturbationStability patterns
        
        Returns:
        --------
        (raw_score, confidence) where raw_score in [0, 1]:
        - 0.0 = strongly authentic (high stability, high curvature)
        - 0.5 = uncertain
        - 1.0 = strongly synthetic (low stability, low curvature)
        """
        # Check feature validity first
        required_features = ['stability_score', 'curvature_score', 'stability_variance']
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > self.params.ZERO_TOLERANCE]
        
        if (len(valid_features) < self.params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return self.params.NEUTRAL_PROBABILITY, self.params.LOW_FEATURE_CONFIDENCE


        # Initialize synthetic_indicator list
        synthetic_indicators    = list()
        
        # Stability Interpretation: Lower = more synthetic
        stability = features['stability_score']
        if (stability < self.params.STABILITY_STRONG_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_STRONG * self.params.STABILITY_WEIGHT)
        
        elif (stability < self.params.STABILITY_MODERATE_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_MODERATE * self.params.STABILITY_WEIGHT)
        
        elif (stability < self.params.STABILITY_WEAK_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_WEAK * self.params.STABILITY_WEIGHT)
        
        elif (stability < self.params.STABILITY_AUTHENTIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_NEUTRAL * self.params.STABILITY_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.PROB_WEIGHT_AUTHENTIC * self.params.STABILITY_WEIGHT)
        
        # Curvature Interpretation: Lower = more synthetic
        curvature = features['curvature_score']
        if (curvature < self.params.CURVATURE_STRONG_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_STRONG * self.params.CURVATURE_WEIGHT)
        
        elif (curvature < self.params.CURVATURE_MODERATE_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_MODERATE * self.params.CURVATURE_WEIGHT)
        
        elif (curvature < self.params.CURVATURE_WEAK_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_WEAK * self.params.CURVATURE_WEIGHT)
        
        elif (curvature < self.params.CURVATURE_AUTHENTIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_NEUTRAL * self.params.CURVATURE_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.PROB_WEIGHT_AUTHENTIC * self.params.CURVATURE_WEIGHT)
        
        # Variance Interpretation: Lower = more synthetic (consistent across chunks)
        variance = features['stability_variance']
        if (variance < self.params.VARIANCE_STRONG_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_STRONG * self.params.VARIANCE_WEIGHT)
        
        elif (variance < self.params.VARIANCE_MODERATE_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_MODERATE * self.params.VARIANCE_WEIGHT)
        
        elif (variance < self.params.VARIANCE_WEAK_SYNTHETIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_WEAK * self.params.VARIANCE_WEIGHT)
        
        elif (variance < self.params.VARIANCE_AUTHENTIC):
            synthetic_indicators.append(self.params.PROB_WEIGHT_NEUTRAL * self.params.VARIANCE_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.PROB_WEIGHT_AUTHENTIC * self.params.VARIANCE_WEIGHT)
        
        # Calculate raw score and confidence
        if synthetic_indicators:
            total_weight            = (self.params.STABILITY_WEIGHT + self.params.CURVATURE_WEIGHT + self.params.VARIANCE_WEIGHT)
            raw_score               = sum(synthetic_indicators) / total_weight
            
            # Confidence: Based on perturbation success and signal agreement
            num_valid               = features.get('num_valid_perturbations', 0)
            perturbation_confidence = min(1.0, num_valid / self.params.NUM_PERTURBATIONS)
            
            # Calculate agreement between signals
            weights                 = [self.params.STABILITY_WEIGHT, self.params.CURVATURE_WEIGHT, self.params.VARIANCE_WEIGHT]
            normalized_indicators   = [si / weights[i] for i, si in enumerate(synthetic_indicators)]
            agreement               = 1.0 - (np.std(normalized_indicators) if len(normalized_indicators) > 1 else 0.5)
            
            confidence = (self.params.CONFIDENCE_BASE + 
                         self.params.CONFIDENCE_PERTURBATION_FACTOR * perturbation_confidence +
                         self.params.CONFIDENCE_AGREEMENT_FACTOR * agreement)
       
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
        stability = features['stability_score']
        if (self.params.STABILITY_MIXED_MIN <= stability <= self.params.STABILITY_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)
       
        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # High stability variance suggests mixed content
        stability_variance = features['stability_variance']
        if (stability_variance > self.params.STABILITY_VARIANCE_HIGH):
            hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)

        elif (stability_variance > self.params.STABILITY_VARIANCE_MEDIUM):
            hybrid_indicators.append(self.params.VERY_WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # Moderate curvature might indicate mixing
        curvature = features['curvature_score']
        if (self.params.CURVATURE_MIXED_MIN <= curvature <= self.params.CURVATURE_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        hybrid_prob = np.mean(hybrid_indicators) if hybrid_indicators else 0.0
        return min(self.params.MAX_HYBRID_PROBABILITY, hybrid_prob)
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return more meaningful default features
        """
        return {"original_log_prob"         : self.params.DEFAULT_ORIGINAL_LOG_PROB,
                "avg_perturbed_log_prob"    : self.params.DEFAULT_AVG_PERTURBED_LOG_PROB,
                "stability_score"           : self.params.DEFAULT_STABILITY_SCORE, 
                "curvature_score"           : self.params.DEFAULT_CURVATURE_SCORE,
                "perturbation_variance"     : self.params.DEFAULT_PERTURBATION_VARIANCE,
                "avg_chunk_stability"       : self.params.DEFAULT_AVG_CHUNK_STABILITY,
                "stability_variance"        : self.params.DEFAULT_STABILITY_VARIANCE,
                "num_perturbations"         : 0,
                "num_valid_perturbations"   : 0,
                "num_chunks_analyzed"       : 0,
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