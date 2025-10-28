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



class DetectGPTMetric(BaseMetric):
    """
    DetectGPT implementation for text stability analysis under perturbations
    
    Measures:
    - Text stability under random perturbations
    - Likelihood curvature analysis
    - Masked token prediction analysis
    """
    def __init__(self):
        super().__init__(name        = "detect_gpt",
                         description = "Text stability analysis under perturbations (DetectGPT method)",
                        )
        
        self.gpt_model      = None
        self.gpt_tokenizer  = None
        self.mask_model     = None
        self.mask_tokenizer = None
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    

    def initialize(self) -> bool:
        """
        Initialize the DetectGPT metric
        """
        try:
            logger.info("Initializing DetectGPT metric...")
            
            # Load GPT-2 model for likelihood calculation
            model_manager = get_model_manager()
            gpt_result    = model_manager.load_model("detectgpt_base")
            
            if isinstance(gpt_result, tuple):
                self.gpt_model, self.gpt_tokenizer = gpt_result
                # Move model to appropriate device
                self.gpt_model.to(self.device)
           
            else:
                logger.error("Failed to load GPT-2 model for DetectGPT")
                return False
            
            # Load masked language model for perturbations
            mask_result = model_manager.load_model("detectgpt_mask")
            
            if (isinstance(mask_result, tuple)):
                self.mask_model, self.mask_tokenizer = mask_result
                # Move model to appropriate device
                self.mask_model.to(self.device)
                
                # Ensure tokenizer has padding token
                if (self.mask_tokenizer.pad_token is None):
                    self.mask_tokenizer.pad_token = self.mask_tokenizer.eos_token or '[PAD]'

            else:
                logger.warning("Failed to load mask model, using GPT-2 only")
            
            self.is_initialized = True
            
            logger.success("DetectGPT metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DetectGPT metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute DetectGPT analysis with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if ((not text) or (len(text.strip()) < 100)):
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for DetectGPT analysis",
                                   )
            
            # Get domain-specific thresholds
            domain               = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds    = get_threshold_for_domain(domain)
            detectgpt_thresholds = domain_thresholds.detect_gpt
            
            # Check if we should run this computationally expensive metric
            if (kwargs.get('skip_expensive', False)):
                logger.info("Skipping DetectGPT due to computational constraints")
                
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.3,
                                    error             = "Skipped for performance",
                                   )
            
            # Calculate DetectGPT features
            features                        = self._calculate_detectgpt_features(text)
            
            # Calculate raw DetectGPT score (0-1 scale)
            raw_detectgpt_score, confidence = self._analyze_detectgpt_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob = self._apply_domain_thresholds(raw_detectgpt_score, detectgpt_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                     *= detectgpt_thresholds.confidence_multiplier
            confidence                      = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : detectgpt_thresholds.ai_threshold,
                                                     'human_threshold' : detectgpt_thresholds.human_threshold,
                                                     'raw_score'       : raw_detectgpt_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in DetectGPT computation: {repr(e)}")

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
    

    def _calculate_detectgpt_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive DetectGPT features
        """
        if not self.gpt_model or not self.gpt_tokenizer:
            return self._get_default_features()
        
        try:
            # Preprocess text for better analysis
            processed_text        = self._preprocess_text_for_analysis(text)
            
            # Calculate original text likelihood
            original_likelihood   = self._calculate_likelihood(processed_text)
            
            # Generate perturbations and calculate perturbed likelihoods
            perturbations         = self._generate_perturbations(processed_text, num_perturbations = 5)
            perturbed_likelihoods = list()
            
            for perturbed_text in perturbations:
                if (perturbed_text and (perturbed_text != processed_text)):
                    likelihood = self._calculate_likelihood(perturbed_text)
                    
                    if (likelihood > 0):
                        perturbed_likelihoods.append(likelihood)
            
            # Calculate stability metrics
            if perturbed_likelihoods:
                stability_score          = self._calculate_stability_score(original_likelihood, perturbed_likelihoods)
                curvature_score          = self._calculate_curvature_score(original_likelihood, perturbed_likelihoods)
                variance_score           = np.var(perturbed_likelihoods) if len(perturbed_likelihoods) > 1 else 0.0
                avg_perturbed_likelihood = np.mean(perturbed_likelihoods)
            
            else:
                stability_score          = 0.5
                curvature_score          = 0.5
                variance_score           = 0.1
                avg_perturbed_likelihood = original_likelihood
            
            # Calculate likelihood ratio
            likelihood_ratio            = original_likelihood / avg_perturbed_likelihood if avg_perturbed_likelihood > 0 else 1.0
            
            # Chunk-based analysis for whole-text understanding
            chunk_stabilities           = self._calculate_chunk_stability(processed_text, chunk_size=150)
            stability_variance          = np.var(chunk_stabilities) if chunk_stabilities else 0.0
            avg_chunk_stability         = np.mean(chunk_stabilities) if chunk_stabilities else stability_score
            
            # Normalize scores to 0-1 range
            normalized_stability        = min(1.0, max(0.0, stability_score))
            normalized_curvature        = min(1.0, max(0.0, curvature_score))
            normalized_likelihood_ratio = min(2.0, likelihood_ratio) / 2.0    # Normalize to 0-1
            
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
            logger.warning(f"DetectGPT feature calculation failed: {repr(e)}")
            return self._get_default_features()
    

    def _calculate_likelihood(self, text: str) -> float:
        """
        Calculate log-likelihood of text using GPT-2 with robust error handling
        """
        try:
            # Check text length before tokenization
            if (len(text.strip()) < 10):
                return 0.0

            # Configure tokenizer for proper padding
            tokenizer      = self._configure_tokenizer_padding(self.gpt_tokenizer)
            
            # Tokenize text with proper settings
            encodings      = tokenizer(text, 
                                       return_tensors = 'pt', 
                                       truncation     = True,
                                       max_length     = 512,
                                       padding        = True,
                                       return_attention_mask = True,
                                      )

            input_ids      = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            # Minimum tokens for meaningful analysis
            if ((input_ids.numel() == 0) or (input_ids.size(1) < 5)):
                return 0.0
            
            # Calculate negative log likelihood
            with torch.no_grad():
                outputs = self.gpt_model(input_ids, 
                                         attention_mask = attention_mask, 
                                         labels         = input_ids,
                                        )

                loss    = outputs.loss
            
            # Convert to positive log likelihood (higher = more likely)
            log_likelihood = -loss.item()

            # Reasonable range check (typical values are between -10 and 10)
            if (abs(log_likelihood) > 100):
                logger.warning(f"Extreme likelihood value detected: {log_likelihood}")
                return 0.0
            
            return log_likelihood
            
        except Exception as e:
            logger.warning(f"Likelihood calculation failed: {repr(e)}")
            return 0.0
    

    def _generate_perturbations(self, text: str, num_perturbations: int = 5) -> List[str]:
        """
        Generate perturbed versions of the text with robust error handling
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
            
            # Method 3: RoBERTa-specific masked word replacement
            if (self.mask_model and self.mask_tokenizer and (len(words) > 4) and len(perturbations) < num_perturbations):
                
                try:
                    roberta_perturbations = self._generate_roberta_masked_perturbations(processed_text, 
                                                                                        words, 
                                                                                        num_perturbations - len(perturbations))
                    perturbations.extend(roberta_perturbations)
                    
                except Exception as e:
                    logger.warning(f"RoBERTa masked perturbation failed: {repr(e)}")
            
            # Method 4: Synonym replacement as fallback
            if (len(perturbations) < num_perturbations):
                try:
                    synonym_perturbations = self._generate_synonym_perturbations(processed_text, 
                                                                                 words, 
                                                                                 num_perturbations - len(perturbations))
                    perturbations.extend(synonym_perturbations)
                    
                except Exception as e:
                    logger.debug(f"Synonym replacement failed: {e}")
            
            # Ensure we have at least some perturbations
            if not perturbations:
                # Fallback: create simple variations
                fallback_perturbations = self._generate_fallback_perturbations(processed_text, words)
                perturbations.extend(fallback_perturbations)
            
            # Remove duplicates and ensure we don't exceed requested number
            unique_perturbations = list()
            
            for p in perturbations:
                if (p and (p != processed_text) and (p not in unique_perturbations) and (self._is_valid_perturbation(p, processed_text))):
                    unique_perturbations.append(p)
            
            return unique_perturbations[:num_perturbations]
            
        except Exception as e:
            logger.warning(f"Perturbation generation failed: {repr(e)}")
            # Return at least the original text as fallback
            return [text]
    

    def _generate_roberta_masked_perturbations(self, text: str, words: List[str], max_perturbations: int) -> List[str]:
        """
        Generate perturbations using RoBERTa mask filling
        """
        perturbations = list()
        
        try:
            # RoBERTa uses <mask> token
            roberta_mask_token  = "<mask>"
            
            # Select words to mask (avoid very short words and punctuation)
            candidate_positions = [i for i, word in enumerate(words) if (len(word) > 3) and word.isalpha() and word.lower() not in ['the', 'and', 'but', 'for', 'with']]
            
            if not candidate_positions:
                candidate_positions = [i for i, word in enumerate(words) if len(word) > 2]
            
            if not candidate_positions:
                return perturbations
            
            # Try multiple mask positions
            attempts          = min(max_perturbations * 2, len(candidate_positions))
            positions_to_try  = np.random.choice(candidate_positions, min(attempts, len(candidate_positions)), replace=False)
            
            for pos in positions_to_try:
                if (len(perturbations) >= max_perturbations):
                    break
                    
                try:
                    # Create masked text
                    masked_words      = words.copy()
                    original_word     = masked_words[pos]
                    masked_words[pos] = roberta_mask_token
                    masked_text       = ' '.join(masked_words)
                    
                    # RoBERTa works better with proper sentence structure
                    if not masked_text.endswith(('.', '!', '?')):
                        masked_text += '.'
                    
                    # Tokenize with RoBERTa-specific settings
                    inputs = self.mask_tokenizer(masked_text,
                                                 return_tensors = "pt",
                                                 truncation     = True,
                                                 max_length     = min(128, self.mask_tokenizer.model_max_length),  # Conservative length
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
                                # Use first valid prediction
                                break  
                    
                except Exception as e:
                    logger.debug(f"RoBERTa mask filling failed for position {pos}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"RoBERTa masked perturbations failed: {e}")
        
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
            logger.debug(f"Synonym replacement failed: {e}")
        
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
            logger.debug(f"Fallback perturbation failed: {e}")
        
        return [p for p in perturbations if p and p != text][:3]
    

    def _calculate_stability_score(self, original_likelihood: float, perturbed_likelihoods: List[float]) -> float:
        """
        Calculate text stability score under perturbations : AI text tends to be less stable (larger likelihood drops)
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= 0)):
            return 0.5
        
        # Calculate average likelihood drop
        likelihood_drops = [(original_likelihood - pl) / original_likelihood for pl in perturbed_likelihoods]
        avg_drop         = np.mean(likelihood_drops) if likelihood_drops else 0.0
        
        # Higher drop = less stable = more AI-like : Normalize to 0-1 scale (assume max drop of 50%)
        stability_score  = min(1.0, avg_drop / 0.5)

        return stability_score
    

    def _calculate_curvature_score(self, original_likelihood: float, perturbed_likelihoods: List[float]) -> float:
        """
        Calculate likelihood curvature score : AI text often has different curvature properties
        """
        if ((not perturbed_likelihoods) or (original_likelihood <= 0)):
            return 0.5
        
        # Calculate variance of likelihood changes
        likelihood_changes = [abs(original_likelihood - pl) for pl in perturbed_likelihoods]
        change_variance    = np.var(likelihood_changes) if len(likelihood_changes) > 1 else 0.0
        
        # Higher variance = more curvature = potentially more AI-like : Normalize based on typical variance ranges
        curvature_score    = min(1.0, change_variance * 10.0)  # Adjust scaling factor as needed
        
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
                    chunk_likelihood = self._calculate_likelihood(chunk)
                    
                    if (chunk_likelihood > 0):
                        # Generate a simple perturbation for this chunk
                        chunk_words = chunk.split()
                        
                        if (len(chunk_words) > 5):
                            # Delete 10% of words
                            delete_count         = max(1, len(chunk_words) // 10)
                            indices_to_keep      = np.random.choice(len(chunk_words), len(chunk_words) - delete_count, replace=False)
                            perturbed_chunk      = ' '.join([chunk_words[i] for i in sorted(indices_to_keep)])
                            
                            perturbed_likelihood = self._calculate_likelihood(perturbed_chunk)

                            if (perturbed_likelihood > 0):
                                stability = (chunk_likelihood - perturbed_likelihood) / chunk_likelihood
                                stabilities.append(min(1.0, max(0.0, stability)))
                except Exception:
                    continue
        
        return stabilities
    

    def _analyze_detectgpt_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze DetectGPT patterns to determine RAW DetectGPT score (0-1 scale) : Higher score = more AI-like
        """
        # Check feature validity first
        required_features = ['stability_score', 'curvature_score', 'normalized_likelihood_ratio', 'stability_variance', 'perturbation_variance']
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > 0]
        
        if (len(valid_features) < 3):
            # Low confidence if insufficient features
            return 0.5, 0.3  


        # Initialize ai_indicator list
        ai_indicators = list()
        
        # High stability score suggests AI (larger likelihood drops)
        if (features['stability_score'] > 0.6):
            ai_indicators.append(0.8)

        elif (features['stability_score'] > 0.3):
            ai_indicators.append(0.5)
        
        else:
            ai_indicators.append(0.2)
        
        # High curvature score suggests AI
        if (features['curvature_score'] > 0.7):
            ai_indicators.append(0.7)

        elif (features['curvature_score'] > 0.4):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # High likelihood ratio suggests AI (original much more likely than perturbations)
        if (features['normalized_likelihood_ratio'] > 0.8):
            ai_indicators.append(0.9)
        
        elif (features['normalized_likelihood_ratio'] > 0.6):
            ai_indicators.append(0.6)

        else:
            ai_indicators.append(0.3)
        
        # Low stability variance suggests AI (consistent across chunks)
        if (features['stability_variance'] < 0.05):
            ai_indicators.append(0.7)

        elif (features['stability_variance'] < 0.1):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # High perturbation variance suggests AI
        if (features['perturbation_variance'] > 0.1):
            ai_indicators.append(0.6)

        elif (features['perturbation_variance'] > 0.05):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # Calculate raw score and confidence
        raw_score  = np.mean(ai_indicators) if ai_indicators else 0.5
        confidence = 1.0 - (np.std(ai_indicators) / 0.5) if ai_indicators else 0.5
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
        Return default features when analysis is not possible
        """
        return {"original_likelihood"         : 2.0,
                "avg_perturbed_likelihood"    : 1.8,
                "likelihood_ratio"            : 1.1,
                "normalized_likelihood_ratio" : 0.55,
                "stability_score"             : 0.5,
                "curvature_score"             : 0.5,
                "perturbation_variance"       : 0.05,
                "avg_chunk_stability"         : 0.5,
                "stability_variance"          : 0.1,
                "num_perturbations"           : 0,
                "num_valid_perturbations"     : 0,
                "num_chunks_analyzed"         : 0,
               }
    

    def _preprocess_text_for_analysis(self, text: str) -> str:
        """
        Preprocess text for DetectGPT analysis
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
        
        # RoBERTa works better with proper punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Truncate to safe length
        if (len(text) > 1000):
            sentences = text.split('. ')
            if len(sentences) > 1:
                # Keep first few sentences
                text = '. '.join(sentences[:3]) + '.'
            
            else:
                text = text[:1000]
        
        return text
    

    def _configure_tokenizer_padding(self, tokenizer) -> Any:
        """
        Configure tokenizer for proper padding
        """
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        tokenizer.padding_side = "left"
        
        return tokenizer
    

    def _clean_roberta_token(self, token: str) -> str:
        """
        Clean tokens from RoBERTa tokenizer
        """
        if not token:
            return ""
        
        # Remove RoBERTa-specific artifacts
        token = token.replace('Ġ', ' ')  # RoBERTa space marker
        token = token.replace('</s>', '')
        token = token.replace('<s>', '')
        token = token.replace('<pad>', '')
        
        # Remove leading/trailing whitespace and punctuation
        token = token.strip(' .,!?;:"\'')
        
        return token
    

    def _is_valid_perturbation(self, perturbed_text: str, original_text: str) -> bool:
        """
        Check if a perturbation is valid
        """
        # Not too short
        return (perturbed_text and 
                len(perturbed_text.strip()) > 10 and 
                perturbed_text != original_text and
                len(perturbed_text) > len(original_text) * 0.5)  
    

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
__all__ = ["DetectGPTMetric"]