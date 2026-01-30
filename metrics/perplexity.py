# DEPENDENCIES
import re
import math
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
from config.constants import perplexity_metric_params
from config.threshold_config import get_threshold_for_domain


class PerplexityMetric(BaseMetric):
    """
    Text predictability analysis using GPT-2 for perplexity calculation
    
    Measures (Aligned with Documentation):
    - Overall text perplexity (lower = more predictable = more synthetic-like)
    - Perplexity distribution across text chunks
    - Sentence-level perplexity patterns
    - Cross-entropy analysis
    
    Mathematical Foundation:
    - Perplexity = exp(cross_entropy_loss)
    - Lower perplexity indicates more predictable text (characteristic of synthetic text)
    - Higher perplexity indicates less predictable text (characteristic of human text)
    """
    def __init__(self):
        super().__init__(name        = "perplexity",
                         description = "GPT-2 based perplexity calculation for text predictability analysis",
                        )

        self.model     = None
        self.tokenizer = None
        self.params    = perplexity_metric_params
    

    def initialize(self) -> bool:
        """
        Initialize the perplexity metric
        """
        try:
            logger.info("Initializing perplexity metric...")
            
            # Load GPT-2 model and tokenizer
            model_manager = get_model_manager()
            model_result  = model_manager.load_model(model_name = "perplexity_reference_lm")
            
            if isinstance(model_result, tuple):
                self.model, self.tokenizer = model_result
            
            else:
                logger.error("Failed to load GPT-2 model for perplexity calculation")
                return False
            
            self.is_initialized = True
            logger.success("Perplexity metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize perplexity metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute perplexity measures with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if (not text or len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_ANALYSIS):
                return MetricResult(metric_name           = self.name,
                                    synthetic_probability = self.params.NEUTRAL_PROBABILITY,
                                    authentic_probability = self.params.NEUTRAL_PROBABILITY,
                                    hybrid_probability    = self.params.MIN_PROBABILITY,
                                    confidence            = self.params.MIN_CONFIDENCE,
                                    error                 = "Text too short for perplexity analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                                       = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                            = get_threshold_for_domain(domain)
            perplexity_thresholds                        = domain_thresholds.perplexity
            
            # Calculate comprehensive perplexity features
            features                                     = self._calculate_perplexity_features(text = text)
            
            # Calculate raw perplexity score (0-1 scale) with IMPROVED confidence
            raw_perplexity_score, confidence             = self._analyze_perplexity_patterns(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob  = self._apply_domain_thresholds(raw_score  = raw_perplexity_score, 
                                                                                         thresholds = perplexity_thresholds, 
                                                                                         features   = features,
                                                                                        )
                                                                                        
            # Apply confidence multiplier from domain thresholds
            confidence                                  *= perplexity_thresholds.confidence_multiplier
            confidence                                   = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
            
            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'        : domain.value,
                                                         'synthetic_threshold': perplexity_thresholds.synthetic_threshold,
                                                         'authentic_threshold': perplexity_thresholds.authentic_threshold,
                                                         'raw_score'          : raw_perplexity_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in perplexity computation: {repr(e)}")
            return self._default_result(error = str(e))
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        synthetic_threshold = thresholds.synthetic_threshold
        authentic_threshold = thresholds.authentic_threshold
        
        # Calculate probabilities based on threshold distances
        if (raw_score >= synthetic_threshold):
            distance       = raw_score - synthetic_threshold
            synthetic_prob = self.params.STRONG_SYNTHETIC_BASE_PROB + distance * self.params.WEAK_PROBABILITY_ADJUSTMENT
            authentic_prob = (self.params.MAX_PROBABILITY - self.params.STRONG_SYNTHETIC_BASE_PROB) - distance * self.params.WEAK_PROBABILITY_ADJUSTMENT

        elif (raw_score <= authentic_threshold):
            distance       = authentic_threshold - raw_score
            synthetic_prob = (self.params.MAX_PROBABILITY - self.params.STRONG_AUTHENTIC_BASE_PROB) - distance * self.params.WEAK_PROBABILITY_ADJUSTMENT
            authentic_prob = self.params.STRONG_AUTHENTIC_BASE_PROB + distance * self.params.WEAK_PROBABILITY_ADJUSTMENT
            
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
        
        # Calculate hybrid probability based on perplexity variance
        hybrid_prob = self._calculate_hybrid_probability(features)
        
        # Normalize to sum to 1.0
        total       = synthetic_prob + authentic_prob + hybrid_prob

        if (total > self.params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob
    

    def _calculate_perplexity_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive perplexity measures
        """
        if not self.model or not self.tokenizer:
            return self._get_default_features()
        
        # Calculate overall perplexity
        overall_perplexity    = self._calculate_perplexity(text)
        
        # Split into sentences for sentence-level analysis
        sentences             = self._split_sentences(text)
        
        # Calculate sentence-level perplexities
        sentence_perplexities = list()
        valid_sentences       = 0
        
        for sentence in sentences:
            # Use constant instead of hardcoded division
            min_sent_len = self.params.MIN_SENTENCE_LENGTH // self.params.MIN_SENTENCE_LENGTH_DIVISOR
            
            if (len(sentence.strip()) > min_sent_len):  
                sent_perplexity = self._calculate_perplexity(sentence)
                
                if (sent_perplexity > self.params.ZERO_TOLERANCE):
                    sentence_perplexities.append(sent_perplexity)
                    valid_sentences += 1
        
        # Calculate statistical features
        if sentence_perplexities:
            avg_sentence_perplexity = np.mean(sentence_perplexities)
            std_sentence_perplexity = np.std(sentence_perplexities)
            min_sentence_perplexity = np.min(sentence_perplexities)
            max_sentence_perplexity = np.max(sentence_perplexities)
        
        else:
            avg_sentence_perplexity = overall_perplexity
            std_sentence_perplexity = 0.0
            min_sentence_perplexity = overall_perplexity
            max_sentence_perplexity = overall_perplexity
        
        # Chunk-based analysis for whole-text understanding
        chunk_perplexities    = self._calculate_chunk_perplexity(text)
        perplexity_variance   = np.var(chunk_perplexities) if chunk_perplexities else 0.0
        avg_chunk_perplexity  = np.mean(chunk_perplexities) if chunk_perplexities else overall_perplexity
        
        # Normalize perplexity to 0-1 scale for easier interpretation
        normalized_perplexity = self._normalize_perplexity(overall_perplexity)
        
        # Cross-entropy analysis
        cross_entropy_score   = self._calculate_cross_entropy(text)
        
        return {"overall_perplexity"      : round(overall_perplexity, 2),
                "normalized_perplexity"   : round(normalized_perplexity, 4),
                "avg_sentence_perplexity" : round(avg_sentence_perplexity, 2),
                "std_sentence_perplexity" : round(std_sentence_perplexity, 2),
                "min_sentence_perplexity" : round(min_sentence_perplexity, 2),
                "max_sentence_perplexity" : round(max_sentence_perplexity, 2),
                "perplexity_variance"     : round(perplexity_variance, 4),
                "avg_chunk_perplexity"    : round(avg_chunk_perplexity, 2),
                "cross_entropy_score"     : round(cross_entropy_score, 4),
                "num_sentences_analyzed"  : valid_sentences,
                "num_chunks_analyzed"     : len(chunk_perplexities),
               }
    

    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for given text using GPT-2
        
        Mathematical Formula:
            perplexity = exp(cross_entropy_loss)
        
        Interpretation:
            Lower perplexity = more predictable = more synthetic-like
            Higher perplexity = less predictable = more human-like
        """
        try:
            # Use constant instead of hardcoded division
            min_text_len = self.params.MIN_SENTENCE_LENGTH // self.params.MIN_SENTENCE_LENGTH_DIVISOR
            
            if (len(text.strip()) < min_text_len):
                return 0.0

            # Tokenize the text
            encodings = self.tokenizer(text, 
                                       return_tensors = 'pt', 
                                       truncation     = True, 
                                       max_length     = self.params.MAX_TOKEN_LENGTH,
                                      )

            input_ids = encodings.input_ids
            
            # Minimum tokens
            if ((input_ids.numel() == 0) or (input_ids.size(1) < self.params.MIN_TOKENS_FOR_PERPLEXITY)):
                return 0.0
            
            # Calculate loss (cross-entropy)
            with torch.no_grad():
                outputs = self.model(input_ids, labels = input_ids)
                loss    = outputs.loss
            
            # Convert loss to perplexity (CORRECT FORMULA)
            perplexity = math.exp(loss.item())

            return perplexity
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {repr(e)}")
            return 0.0
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Use constant instead of hardcoded division
        min_sent_len = self.params.MIN_SENTENCE_LENGTH // self.params.MIN_SENTENCE_LENGTH_DIVISOR
        
        sentences    = re.split(self.params.SENTENCE_SPLIT_PATTERN, text)
        
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > min_sent_len]
    

    def _calculate_chunk_perplexity(self, text: str) -> List[float]:
        """
        Calculate perplexity across text chunks for whole-text analysis
        """
        chunks = list()
        words  = text.split()
        chunk_size = self.params.CHUNK_SIZE_WORDS
        overlap = int(chunk_size * self.params.CHUNK_OVERLAP_RATIO)

        # Use constant instead of hardcoded division
        min_chunk_size = chunk_size // self.params.MIN_CHUNK_SIZE_DIVISOR
        
        if (len(words) < min_chunk_size):
            return [self._calculate_perplexity(text)] if text.strip() else []
        
        # Create overlapping chunks for better analysis
        step = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > self.params.MIN_CHUNK_LENGTH):  
                perplexity = self._calculate_perplexity(chunk)

                # Reasonable range check
                if ((perplexity > self.params.ZERO_TOLERANCE) and (perplexity < self.params.LARGE_PERPLEXITY_THRESHOLD)):
                    chunks.append(perplexity)
        
        # Zero perplexity is physically impossible and biases the score hence returning DEFAULT_OVERALL_PERPLEXITY
        return chunks if chunks else [self.params.DEFAULT_OVERALL_PERPLEXITY]
    

    def _normalize_perplexity(self, perplexity: float) -> float:
        """
        Normalize perplexity using sigmoid transformation
        
        Formula:
            normalized = 1 / (1 + exp((perplexity - center) / scale))
        
        Interpretation:
            Lower perplexity → higher normalized score → more synthetic-like
            Higher perplexity → lower normalized score → more authentic-like
        
        Example transformations:
            perplexity=20  → normalized≈0.73 (high score = synthetic)
            perplexity=40  → normalized≈0.50 (neutral)
            perplexity=80  → normalized≈0.12 (low score = authentic)
        """
        # Use sigmoid normalization (CORRECT IMPLEMENTATION)
        normalized = 1.0 / (1.0 + np.exp((perplexity - self.params.PERPLEXITY_SIGMOID_CENTER) / self.params.PERPLEXITY_SIGMOID_SCALE))
        
        return normalized 
    

    def _calculate_cross_entropy(self, text: str) -> float:
        """
        Calculate cross-entropy as an alternative measure
        """
        try:
            encodings = self.tokenizer(text, 
                                       return_tensors = 'pt', 
                                       truncation     = True, 
                                       max_length     = self.params.MAX_TOKEN_LENGTH)
            input_ids = encodings.input_ids
            
            if (input_ids.numel() == 0):
                return 0.0
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels = input_ids)
                loss    = outputs.loss
            
            # Normalize cross-entropy to 0-1 scale
            cross_entropy = loss.item()
            normalized_ce = min(1.0, cross_entropy / self.params.MAX_CROSS_ENTROPY) 
            
            return normalized_ce
            
        except Exception as e:
            logger.warning(f"Cross-entropy calculation failed: {repr(e)}")
            return 0.0
    

    def _analyze_perplexity_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze perplexity patterns to determine RAW perplexity score (0-1 scale) with IMPROVED confidence
        
        Returns:
            (raw_score, confidence) where:
            - raw_score: Higher score = more synthetic-like
            - confidence: Based on sample size and agreement between indicators
        """
        # Check feature validity first
        required_features = ['normalized_perplexity', 'perplexity_variance', 'std_sentence_perplexity', 'cross_entropy_score']
    
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > self.params.ZERO_TOLERANCE]
    
        if (len(valid_features) < self.params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return self.params.NEUTRAL_PROBABILITY, self.params.LOW_FEATURE_CONFIDENCE


        # Initialize synthetic_indicator list
        synthetic_indicators = list()
        
        # Low overall perplexity suggests synthetic
        if (features['normalized_perplexity'] > self.params.NORMALIZED_PERPLEXITY_HIGH_THRESHOLD):
            # Very synthetic-like
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)  

        elif (features['normalized_perplexity'] > self.params.NORMALIZED_PERPLEXITY_MEDIUM_THRESHOLD):
            # synthetic-like
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)  

        else:
            # authentic-like
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT) 
        
        # Low perplexity variance suggests synthetic (consistent predictability)
        if (features['perplexity_variance'] < self.params.PERPLEXITY_VARIANCE_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        elif (features['perplexity_variance'] < self.params.PERPLEXITY_VARIANCE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Low sentence perplexity std suggests synthetic (consistent across sentences)
        if (features['std_sentence_perplexity'] < self.params.STD_SENTENCE_PERPLEXITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)

        elif (features['std_sentence_perplexity'] < self.params.STD_SENTENCE_PERPLEXITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Low cross-entropy suggests synthetic (more predictable)
        if (features['cross_entropy_score'] < self.params.CROSS_ENTROPY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)
        
        elif (features['cross_entropy_score'] < self.params.CROSS_ENTROPY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Consistent chunk perplexity suggests synthetic
        chunk_variance = features['perplexity_variance']
        
        if (chunk_variance < self.params.CHUNK_VARIANCE_VERY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)

        elif (chunk_variance < self.params.CHUNK_VARIANCE_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.VERY_WEAK_SYNTHETIC_WEIGHT)
        
        # Calculate raw score
        raw_score  = np.mean(synthetic_indicators) if synthetic_indicators else self.params.NEUTRAL_PROBABILITY
        
        # IMPROVED CONFIDENCE CALCULATION
        # Factor 1: Agreement between indicators (lower std = higher confidence)
        agreement_confidence = 1.0 - min(1.0, np.std(synthetic_indicators) / self.params.CONFIDENCE_STD_NORMALIZER)
        
        # Factor 2: Sample size adequacy
        num_sentences       = features.get('num_sentences_analyzed', 0)
        num_chunks          = features.get('num_chunks_analyzed', 0)
        sentence_confidence = min(1.0, num_sentences / self.params.MIN_SENTENCES_FOR_CONFIDENCE)
        chunk_confidence    = min(1.0, num_chunks / self.params.MIN_CHUNKS_FOR_CONFIDENCE)
        sample_confidence   = (sentence_confidence + chunk_confidence) / 2.0
        
        # Combine factors
        confidence = (self.params.CONFIDENCE_BASE + 
                     self.params.CONFIDENCE_STD_FACTOR * agreement_confidence +
                     self.params.CONFIDENCE_SAMPLE_FACTOR * sample_confidence)
        
        confidence = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content
        """
        hybrid_indicators = list()
        
        # Moderate perplexity values might indicate mixing
        if (self.params.NORMALIZED_PERPLEXITY_MIXED_MIN <= features['normalized_perplexity'] <= self.params.NORMALIZED_PERPLEXITY_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # High perplexity variance suggests mixed content
        if (features['perplexity_variance'] > self.params.PERPLEXITY_VARIANCE_HIGH_THRESHOLD):
            hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)

        elif (features['perplexity_variance'] > self.params.PERPLEXITY_VARIANCE_MEDIUM_THRESHOLD):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # Inconsistent sentence perplexities
        if (self.params.STD_SENTENCE_PERPLEXITY_MIXED_MIN <= features['std_sentence_perplexity'] <= self.params.STD_SENTENCE_PERPLEXITY_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        hybrid_prob = np.mean(hybrid_indicators) if hybrid_indicators else 0.0
        return min(self.params.MAX_HYBRID_PROBABILITY, hybrid_prob)
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        return {"overall_perplexity"      : self.params.DEFAULT_OVERALL_PERPLEXITY,
                "normalized_perplexity"   : self.params.DEFAULT_NORMALIZED_PERPLEXITY,
                "avg_sentence_perplexity" : self.params.DEFAULT_AVG_SENTENCE_PERPLEXITY,
                "std_sentence_perplexity" : self.params.DEFAULT_STD_SENTENCE_PERPLEXITY,
                "min_sentence_perplexity" : self.params.DEFAULT_MIN_SENTENCE_PERPLEXITY,
                "max_sentence_perplexity" : self.params.DEFAULT_MAX_SENTENCE_PERPLEXITY,
                "perplexity_variance"     : self.params.DEFAULT_PERPLEXITY_VARIANCE,
                "avg_chunk_perplexity"    : self.params.DEFAULT_AVG_CHUNK_PERPLEXITY,
                "cross_entropy_score"     : self.params.DEFAULT_CROSS_ENTROPY_SCORE,
                "num_sentences_analyzed"  : 0,
                "num_chunks_analyzed"     : 0,
               }
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.model     = None
        self.tokenizer = None
        super().cleanup()



# Export
__all__ = ["PerplexityMetric"]