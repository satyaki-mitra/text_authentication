# DEPENDENCIES
import math
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from collections import Counter
from config.enums import Domain
from config.schemas import MetricResult
from metrics.base_metric import BaseMetric
from models.model_manager import get_model_manager
from config.constants import entropy_metric_params
from config.threshold_config import get_threshold_for_domain


class EntropyMetric(BaseMetric):
    """
    Entropy analysis for text randomness and predictability

    Mathematical Foundation:
    ------------------------
    - Shannon Entropy: H = -Σ p(x) * log2(p(x))
    - Higher entropy = more random/diverse = typically human
    - Lower entropy  = more predictable/uniform = typically synthetic
    
    Measures:
    ---------
    - Character-level entropy and diversity
    - Word-level entropy and burstiness  
    - Token-level diversity and unpredictability in sequences
    - Entropy distribution across text chunks
    - Synthetic-specific pattern detection
    """
    def __init__(self):
        super().__init__(name        = "entropy",
                         description = "Token-level diversity and unpredictability in text sequences",
                        )
        self.tokenizer = None
        self.params    = entropy_metric_params
    

    def initialize(self) -> bool:
        """
        Initialize the entropy metric
        """
        try:
            logger.info("Initializing entropy metric...")
            
            # Load tokenizer for token-level analysis
            model_manager = get_model_manager()
            gpt_model     = model_manager.load_model("perplexity_reference_lm")
            
            if isinstance(gpt_model, tuple):
                self.tokenizer = gpt_model[1]

            else:
                logger.warning("Could not get tokenizer, using character-level entropy only")
            
            self.is_initialized = True
            logger.success("Entropy metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize entropy metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute entropy measures for text
        """
        try:
            if (not text or (len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_ANALYSIS)):
                return MetricResult(metric_name           = self.name,
                                    synthetic_probability = self.params.NEUTRAL_PROBABILITY,
                                    authentic_probability = self.params.NEUTRAL_PROBABILITY,
                                    hybrid_probability    = self.params.MIN_PROBABILITY,
                                    confidence            = self.params.MIN_CONFIDENCE,
                                    error                 = "Text too short for entropy analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                                      = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                           = get_threshold_for_domain(domain)
            entropy_thresholds                          = domain_thresholds.entropy
            
            # Calculate comprehensive entropy features
            features                                    = self._calculate_entropy_features(text = text)
            
            # Calculate raw entropy score (0-1 scale)
            raw_entropy_score, confidence               = self._analyze_entropy_patterns(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob = self._apply_domain_thresholds(raw_score  = raw_entropy_score, 
                                                                                        thresholds = entropy_thresholds, 
                                                                                        features   = features,
                                                                                       )
            
            # Apply confidence multiplier from domain thresholds
            confidence                                 *= entropy_thresholds.confidence_multiplier
            confidence                                  = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
            
            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'        : domain.value,
                                                         'synthetic_threshold': entropy_thresholds.synthetic_threshold,
                                                         'authentic_threshold': entropy_thresholds.authentic_threshold,
                                                         'raw_score'          : raw_entropy_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in entropy computation: {repr(e)}")
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
        
        # Calculate hybrid probability based on entropy variance
        hybrid_prob    = self._calculate_hybrid_probability(features)
        
        # Normalize to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob

        if (total > self.params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob
    

    def _calculate_entropy_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive entropy measures including document-required features
        """
        # Basic entropy measures
        char_entropy              = self._calculate_character_entropy(text = text)
        word_entropy              = self._calculate_word_entropy(text = text)
        token_entropy             = self._calculate_token_entropy(text = text) if self.tokenizer else 0.0
        
        # DOCUMENT-REQUIRED: Token-level diversity
        token_diversity           = self._calculate_token_diversity(text = text)
        
        # DOCUMENT-REQUIRED: Unpredictability in sequences
        sequence_unpredictability = self._calculate_sequence_unpredictability(text = text)
        
        # Chunk-based analysis for whole-text understanding
        chunk_entropies           = self._calculate_chunk_entropy(text = text)
        entropy_variance          = np.var(chunk_entropies) if chunk_entropies else 0.0
        avg_chunk_entropy         = np.mean(chunk_entropies) if chunk_entropies else 0.0
        
        # Synthetic-specific pattern detection
        synthetic_pattern_score   = self._detect_synthetic_entropy_patterns(text = text)
        
        # Predictability measures
        predictability            = 1.0 - min(1.0, char_entropy / self.params.MAX_CHAR_ENTROPY)
        
        # Count tokens for confidence calculation
        num_tokens                = len(self.tokenizer.encode(text, add_special_tokens = False)) if self.tokenizer else len(text.split())
        
        return {"char_entropy"              : round(char_entropy, 4),
                "word_entropy"              : round(word_entropy, 4),
                "token_entropy"             : round(token_entropy, 4),
                "token_diversity"           : round(token_diversity, 4), 
                "sequence_unpredictability" : round(sequence_unpredictability, 4),  
                "entropy_variance"          : round(entropy_variance, 4),
                "avg_chunk_entropy"         : round(avg_chunk_entropy, 4),
                "predictability_score"      : round(predictability, 4),
                "synthetic_pattern_score"   : round(synthetic_pattern_score, 4),
                "num_chunks_analyzed"       : len(chunk_entropies),
                "num_tokens_analyzed"       : num_tokens,
               }
    

    def _calculate_character_entropy(self, text: str) -> float:
        """
        Calculate character-level Shannon entropy
        
        Formula: H = -Σ p(x) * log2(p(x))
        
        Typical English text: 3.0-4.5 bits
        """
        # Clean text and convert to lowercase
        clean_text = ''.join(c for c in text.lower() if c.isalnum() or c.isspace())
        
        if not clean_text:
            return 0.0
        
        # Count character frequencies
        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        
        # Calculate Shannon entropy
        entropy    = 0.0

        for count in char_counts.values():
            probability = count / total_chars

            if (probability > self.params.ZERO_TOLERANCE):
                entropy -= probability * math.log2(probability)
        
        return entropy
    

    def _calculate_word_entropy(self, text: str) -> float:
        """
        Calculate word-level Shannon entropy
        """
        words = text.lower().split()
        if (len(words) < self.params.MIN_WORDS_FOR_ANALYSIS):
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy     = 0.0

        for count in word_counts.values():
            probability = count / total_words

            if (probability > self.params.ZERO_TOLERANCE):
                entropy -= probability * math.log2(probability)
        
        return entropy
    

    def _calculate_token_entropy(self, text: str) -> float:
        """
        Calculate token-level Shannon entropy using GPT-2 tokenizer
        """
        try:
            if not self.tokenizer:
                return 0.0
            
            # Length check before tokenization
            if (len(text.strip()) < self.params.MIN_SENTENCE_LENGTH):
                return 0.0

            # Tokenize text
            tokens = self.tokenizer.encode(text, 
                                           add_special_tokens = False, 
                                           truncation         = True, 
                                          )
            
            if (len(tokens) < self.params.MIN_TOKENS_FOR_ANALYSIS):
                return 0.0
            
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            entropy      = 0.0

            for count in token_counts.values():
                probability = count / total_tokens
                if (probability > self.params.ZERO_TOLERANCE):
                    entropy -= probability * math.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Token entropy calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_token_diversity(self, text: str) -> float:
        """
        Calculate token-level diversity (type-token ratio)
        
        Interpretation:
        --------------
            Higher diversity = more authentic-like
            Lower diversity  = more synthetic-like (vocabulary reuse)
        """
        if not self.tokenizer:
            return 0.0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens = False)
            if (len(tokens) < self.params.MIN_TOKENS_FOR_ANALYSIS):
                return 0.0
            
            unique_tokens = len(set(tokens))
            total_tokens  = len(tokens)
            
            # Type-token ratio
            diversity     = unique_tokens / total_tokens

            return diversity
            
        except Exception as e:
            logger.warning(f"Token diversity calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_sequence_unpredictability(self, text: str) -> float:
        """
        Calculate unpredictability in text sequences using bigram entropy
        
        Measures how unpredictable the token sequences are: Higher unpredictability = more human-like
        """
        if not self.tokenizer:
            return 0.0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens = False)
            if (len(tokens) < self.params.MIN_TOKENS_FOR_SEQUENCE):
                return 0.0
            
            # Calculate bigram unpredictability
            bigrams          = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            bigram_counts    = Counter(bigrams)
            total_bigrams    = len(bigrams)
            
            # Higher entropy = more unpredictable sequences
            sequence_entropy = 0.0

            for count in bigram_counts.values():
                probability = count / total_bigrams
                if (probability > self.params.ZERO_TOLERANCE):
                    sequence_entropy -= probability * math.log2(probability)
            
            # Normalize to 0-1 scale
            normalized_entropy = min(1.0, sequence_entropy / self.params.MAX_BIGRAM_ENTROPY)  
            
            return normalized_entropy
            
        except Exception as e:
            logger.warning(f"Sequence unpredictability calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_chunk_entropy(self, text: str) -> List[float]:
        """
        Calculate entropy distribution across text chunks
        """
        chunks     = list()
        words      = text.split()
        chunk_size = self.params.CHUNK_SIZE_WORDS
        overlap    = int(chunk_size * self.params.CHUNK_OVERLAP_RATIO)
        step       = max(1, chunk_size - overlap)
        
        # Create overlapping chunks for better analysis
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > self.params.MIN_CHUNK_LENGTH):  
                entropy = self._calculate_character_entropy(text = chunk)
                
                if (entropy > self.params.ZERO_TOLERANCE):
                    chunks.append(entropy)
        
        return chunks
    

    def _detect_synthetic_entropy_patterns(self, text: str) -> float:
        """
        Detect synthetic-specific entropy patterns
        
        Synthetic text often shows specific entropy signatures:
        - Overly consistent character distribution
        - Low token diversity
        - Predictable sequences
        - Low entropy variance across chunks
        """
        patterns_detected = 0
        total_patterns    = 4
        
        # Pattern 1: Overly consistent character distribution
        char_entropy      = self._calculate_character_entropy(text = text)
        
        # Synthetic tends to be more consistent 
        if (char_entropy < self.params.CHAR_ENTROPY_LOW_THRESHOLD):  
            patterns_detected += 1
        
        # Pattern 2: Low token diversity
        token_diversity = self._calculate_token_diversity(text)

        # Synthetic reuses tokens more 
        if (token_diversity < self.params.TOKEN_DIVERSITY_MEDIUM_THRESHOLD):  
            patterns_detected += 1
        
        # Pattern 3: Predictable sequences
        sequence_unpredictability = self._calculate_sequence_unpredictability(text = text)
        
        # Synthetic sequences are more predictable
        if (sequence_unpredictability < self.params.SEQUENCE_UNPREDICTABILITY_MEDIUM_THRESHOLD):  
            patterns_detected += 1
        
        # Pattern 4: Low entropy variance across chunks
        chunk_entropies  = self._calculate_chunk_entropy(text = text)
        entropy_variance = np.var(chunk_entropies) if chunk_entropies else 0.0
        
        # Synthetic maintains consistent entropy 
        if (entropy_variance < self.params.ENTROPY_VARIANCE_LOW_THRESHOLD):  
            patterns_detected += 1
        
        return patterns_detected / total_patterns
    

    def _analyze_entropy_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze entropy patterns to determine RAW entropy score (0-1 scale) with confidence
        
        Returns:
        --------
        (raw_score, confidence) where:
        - raw_score: Higher = more synthetic-like
        - confidence: Based on sample size and agreement
        """
        # Check feature validity
        valid_features = [score for score in [features.get('char_entropy', 0),
                                              features.get('token_entropy', 0),
                                              features.get('token_diversity', 0), 
                                              features.get('sequence_unpredictability', 0),
                                              features.get('synthetic_pattern_score', 0)
                                             ] if score > self.params.ZERO_TOLERANCE
                         ]
        
        if (len(valid_features) < self.params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return self.params.NEUTRAL_PROBABILITY, self.params.LOW_FEATURE_CONFIDENCE

        synthetic_indicators = list()
        
        # Synthetic text often has lower character entropy (more predictable)
        if (features['char_entropy'] < self.params.CHAR_ENTROPY_VERY_LOW_THRESHOLD):
            # Strong synthetic indicator
            synthetic_indicators.append(self.params.VERY_STRONG_SYNTHETIC_WEIGHT)  

        elif (features['char_entropy'] < self.params.CHAR_ENTROPY_LOW_THRESHOLD):
            # Moderate synthetic indicator
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)  

        else:
            # Weak synthetic indicator
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT) 

        # Low token entropy suggests synthetic (limited vocabulary reuse)
        if (features['token_entropy'] < self.params.TOKEN_ENTROPY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT) 

                
        # Low entropy variance suggests synthetic (consistent patterns)
        if (features['entropy_variance'] < self.params.ENTROPY_VARIANCE_VERY_LOW_THRESHOLD):
            # Very strong synthetic indicator
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)  

        elif (features['entropy_variance'] < self.params.ENTROPY_VARIANCE_MEDIUM_THRESHOLD):
            # Neutral
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)  

        else:
            # Strong authentic indicator
            synthetic_indicators.append(self.params.VERY_LOW_SYNTHETIC_WEIGHT)  
        
        # Low token diversity suggests synthetic
        if (features['token_diversity'] < self.params.TOKEN_DIVERSITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        elif (features['token_diversity'] < self.params.TOKEN_DIVERSITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.VERY_WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Low sequence unpredictability suggests synthetic
        if (features['sequence_unpredictability'] < self.params.SEQUENCE_UNPREDICTABILITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.VERY_STRONG_SYNTHETIC_WEIGHT)

        elif (features['sequence_unpredictability'] < self.params.SEQUENCE_UNPREDICTABILITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # High synthetic pattern score suggests synthetic
        if (features['synthetic_pattern_score'] > self.params.SYNTHETIC_PATTERN_SCORE_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)
        
        elif (features['synthetic_pattern_score'] > self.params.SYNTHETIC_PATTERN_SCORE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.LOW_SYNTHETIC_WEIGHT)
        
        # Calculate raw score
        raw_score            = np.mean(synthetic_indicators) if synthetic_indicators else self.params.NEUTRAL_PROBABILITY
        
        # Factor 1: Agreement between indicators (lower std = higher confidence)
        agreement_confidence = 1.0 - min(1.0, np.std(synthetic_indicators) / self.params.CONFIDENCE_STD_NORMALIZER)
        
        # Factor 2: Sample size adequacy
        num_chunks           = features.get('num_chunks_analyzed', 0)
        num_tokens           = features.get('num_tokens_analyzed', 0)
        chunk_confidence     = min(1.0, num_chunks / self.params.MIN_CHUNKS_FOR_CONFIDENCE)
        token_confidence     = min(1.0, num_tokens / self.params.MIN_TOKENS_FOR_CONFIDENCE)
        sample_confidence    = (chunk_confidence + token_confidence) / 2.0
        
        # Combine factors
        confidence           = (self.params.CONFIDENCE_BASE + self.params.CONFIDENCE_STD_FACTOR * agreement_confidence + self.params.CONFIDENCE_SAMPLE_FACTOR * sample_confidence)
        
        confidence           = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content with better indicators
        """
        hybrid_indicators = list()
        
        # High entropy variance suggests mixed content
        entropy_variance  = features.get('entropy_variance', 0)
        
        if (entropy_variance > self.params.ENTROPY_VARIANCE_HIGH_THRESHOLD):
            # Strong mixed indicator
            hybrid_indicators.append(self.params.STRONG_HYBRID_WEIGHT)  

        elif (entropy_variance > self.params.ENTROPY_VARIANCE_MIXED_THRESHOLD):
            hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # Inconsistent patterns across different entropy measures
        char_entropy = features.get('char_entropy', 0)
        word_entropy = features.get('word_entropy', 0)
        
        if ((char_entropy > self.params.ZERO_TOLERANCE) and (word_entropy > self.params.ZERO_TOLERANCE)):
            entropy_discrepancy = abs(char_entropy - word_entropy)

            # Large discrepancy suggests mixing
            if (entropy_discrepancy > self.params.ENTROPY_DISCREPANCY_THRESHOLD):  
                hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)
        
        # Moderate synthetic pattern score might indicate mixing
        synthetic_pattern_score = features.get('synthetic_pattern_score', 0)
        if (self.params.SYNTHETIC_PATTERN_MIXED_MIN <= synthetic_pattern_score <= self.params.SYNTHETIC_PATTERN_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)
        
        hybrid_probability = min(self.params.MAX_HYBRID_PROBABILITY, np.mean(hybrid_indicators)) if hybrid_indicators else 0.0
        
        return hybrid_probability


    def cleanup(self):
        """
        Clean up resources
        """
        self.tokenizer = None
        super().cleanup()



# Export
__all__ = ["EntropyMetric"]