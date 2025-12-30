# DEPENDENCIES
import re
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from collections import Counter
from config.enums import Domain
from config.schemas import MetricResult
from metrics.base_metric import StatisticalMetric 
from config.constants import structural_metric_params
from config.threshold_config import get_threshold_for_domain


class StructuralMetric(StatisticalMetric):
    """
    Structural analysis of text patterns with domain-aware thresholds
    
    Analyzes various structural features including:
    - Sentence length distribution and variance
    - Word length distribution  
    - Punctuation patterns
    - Vocabulary richness
    - Burstiness (variation in patterns)
    """
    def __init__(self):
        super().__init__(name        = "structural",
                         description = "Structural and pattern analysis of the text",
                        )
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute structural features with domain aware thresholds
        
        Arguments:
        ----------
            text     { str } : Input text to analyze

            **kwargs         : Additional parameters including 'domain'
            
        Returns:
        --------
            { MetricResult } : MetricResult with synthetic/authentic probabilities
        """
        try:
            # Get domain-specific thresholds
            domain                                      = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                           = get_threshold_for_domain(domain)
            structural_thresholds                       = domain_thresholds.structural
            
            # Extract all structural features
            features                                    = self._extract_features(text = text)
            
            # Calculate raw synthetic probability based on features
            raw_synthetic_score, confidence             = self._calculate_synthetic_probability(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob = self._apply_domain_thresholds(raw_score  = raw_synthetic_score, 
                                                                                        thresholds = structural_thresholds, 
                                                                                        features   = features,
                                                                                       )
            
            # Apply confidence multiplier from domain thresholds
            confidence                                 *= structural_thresholds.confidence_multiplier
            confidence                                  = max(structural_metric_params.MIN_CONFIDENCE, min(structural_metric_params.MAX_CONFIDENCE, confidence))

            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'        : domain.value,
                                                         'synthetic_threshold': structural_thresholds.synthetic_threshold,
                                                         'authentic_threshold': structural_thresholds.authentic_threshold,
                                                         'raw_score'          : raw_synthetic_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in {self.name} computation: {repr(e)}")
            return self._default_result(error = str(e))
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        params              = structural_metric_params
        synthetic_threshold = thresholds.synthetic_threshold
        authentic_threshold = thresholds.authentic_threshold
        
        # Calculate probabilities based on threshold distances
        if (raw_score >= synthetic_threshold):
            # Above synthetic threshold - strongly synthetic
            distance_from_threshold = raw_score - synthetic_threshold
            synthetic_prob          = params.STRONG_SYNTHETIC_BASE_PROB + (distance_from_threshold * params.WEAK_PROBABILITY_ADJUSTMENT)
            authentic_prob          = (params.MAX_PROBABILITY - params.STRONG_SYNTHETIC_BASE_PROB) - (distance_from_threshold * params.WEAK_PROBABILITY_ADJUSTMENT)
        
        elif (raw_score <= authentic_threshold):
            # Below authentic threshold - strongly authentic
            distance_from_threshold = authentic_threshold - raw_score
            synthetic_prob          = (params.MAX_PROBABILITY - params.STRONG_AUTHENTIC_BASE_PROB) - (distance_from_threshold * params.WEAK_PROBABILITY_ADJUSTMENT)
            authentic_prob          = params.STRONG_AUTHENTIC_BASE_PROB + (distance_from_threshold * params.WEAK_PROBABILITY_ADJUSTMENT)
        
        else:
            # Between thresholds - uncertain zone
            range_width = synthetic_threshold - authentic_threshold
            
            if (range_width > params.ZERO_TOLERANCE):
                position_in_range = (raw_score - authentic_threshold) / range_width
                synthetic_prob    = params.UNCERTAIN_SYNTHETIC_RANGE_START + (position_in_range * params.UNCERTAIN_RANGE_WIDTH)
                authentic_prob    = params.UNCERTAIN_AUTHENTIC_RANGE_START - (position_in_range * params.UNCERTAIN_RANGE_WIDTH)
            
            else:
                synthetic_prob = params.NEUTRAL_PROBABILITY
                authentic_prob = params.NEUTRAL_PROBABILITY
        
        # Ensure probabilities are valid
        synthetic_prob = max(params.MIN_PROBABILITY, min(params.MAX_PROBABILITY, synthetic_prob))
        authentic_prob = max(params.MIN_PROBABILITY, min(params.MAX_PROBABILITY, authentic_prob))
        
        # Calculate hybrid probability based on statistical patterns
        hybrid_prob    = self._calculate_hybrid_probability(features = features)
        
        # Normalize to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob
        
        if (total > params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob

    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract all structural features from text
        """
        # Basic tokenization
        sentences           = self._split_sentences(text = text)
        words               = self._tokenize_words(text = text)
        
        # Sentence-level features
        sentence_lengths    = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else structural_metric_params.ZERO_VALUE
        std_sentence_length = np.std(sentence_lengths) if len(sentence_lengths) > structural_metric_params.MIN_SENTENCE_LENGTH_FOR_STD else structural_metric_params.ZERO_VALUE
        
        # Word-level features
        word_lengths        = [len(w) for w in words]
        avg_word_length     = np.mean(word_lengths) if word_lengths else structural_metric_params.ZERO_VALUE
        std_word_length     = np.std(word_lengths) if len(word_lengths) > structural_metric_params.MIN_WORD_LENGTH_FOR_STD else structural_metric_params.ZERO_VALUE
        
        # Vocabulary richness
        vocabulary_size     = len(set(words))
        type_token_ratio    = vocabulary_size / len(words) if words else structural_metric_params.ZERO_VALUE
        
        # Punctuation analysis
        punctuation_density = self._calculate_punctuation_density(text = text)
        comma_frequency     = text.count(',') / len(words) if words else structural_metric_params.ZERO_VALUE
        
        # Burstiness (variation in patterns)
        burstiness          = self._calculate_burstiness(values = sentence_lengths)
        
        # Uniformity scores
        if (avg_sentence_length > structural_metric_params.ZERO_TOLERANCE):
            length_uniformity = structural_metric_params.MAX_PROBABILITY - (std_sentence_length / avg_sentence_length)
            length_uniformity = max(structural_metric_params.MIN_PROBABILITY, min(structural_metric_params.MAX_PROBABILITY, length_uniformity))
        
        else:
            length_uniformity = structural_metric_params.MIN_PROBABILITY
        
        # Readability approximation (simplified)
        readability         = self._calculate_readability(text      = text, 
                                                          sentences = sentences,
                                                          words     =  words,
                                                         )
        
        # Pattern detection
        repetition_score    = self._detect_repetitive_patterns(words = words)
        
        # N-gram analysis
        bigram_diversity    = self._calculate_ngram_diversity(words = words, 
                                                              n     = structural_metric_params.BIGRAM_N,
                                                             )

        trigram_diversity   = self._calculate_ngram_diversity(words = words, 
                                                              n     = structural_metric_params.TRIGRAM_N,
                                                             )
        
        return {"avg_sentence_length" : round(avg_sentence_length, 2),
                "std_sentence_length" : round(std_sentence_length, 2),
                "avg_word_length"     : round(avg_word_length, 2),
                "std_word_length"     : round(std_word_length, 2),
                "vocabulary_size"     : vocabulary_size,
                "type_token_ratio"    : round(type_token_ratio, 4),
                "punctuation_density" : round(punctuation_density, 4),
                "comma_frequency"     : round(comma_frequency, 4),
                "burstiness_score"    : round(burstiness, 4),
                "length_uniformity"   : round(length_uniformity, 4),
                "readability_score"   : round(readability, 2),
                "repetition_score"    : round(repetition_score, 4),
                "bigram_diversity"    : round(bigram_diversity, 4),
                "trigram_diversity"   : round(trigram_diversity, 4),
                "num_sentences"       : len(sentences),
                "num_words"           : len(words),
               }

    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        sentences = re.split(structural_metric_params.SENTENCE_SPLIT_PATTERN, text)
        
        return [s.strip() for s in sentences if s.strip()]
    

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words
        """
        words = re.findall(structural_metric_params.WORD_TOKENIZE_PATTERN, text.lower())
        
        return words
    

    def _calculate_punctuation_density(self, text: str) -> float:
        """
        Calculate punctuation density
        """
        punctuation = re.findall(structural_metric_params.PUNCTUATION_PATTERN, text)
        total_chars = len(text)
        
        return len(punctuation) / total_chars if total_chars > structural_metric_params.ZERO_TOLERANCE else structural_metric_params.ZERO_VALUE
    

    def _calculate_burstiness(self, values: List[float]) -> float:
        """
        Calculate burstiness score (variation in patterns): Higher burstiness typically indicates human writing
        """
        if (len(values) < structural_metric_params.MIN_VALUES_FOR_BURSTINESS):
            return structural_metric_params.ZERO_VALUE
        
        mean_val   = np.mean(values)
        std_val    = np.std(values)
        
        if (mean_val < structural_metric_params.ZERO_TOLERANCE):
            return structural_metric_params.ZERO_VALUE
        
        # Coefficient of variation
        cv         = std_val / mean_val
        
        # Normalize to 0-1 range
        burstiness = min(structural_metric_params.MAX_PROBABILITY, cv / structural_metric_params.BURSTINESS_NORMALIZATION_FACTOR)
        
        return burstiness
    

    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> float:
        """
        Calculate simplified readability score: Approximation of Flesch Reading Ease
        """
        if not sentences or not words:
            return structural_metric_params.NEUTRAL_READABILITY_SCORE
        
        total_sentences = len(sentences)
        total_words     = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease approximation
        if ((total_sentences > structural_metric_params.ZERO_TOLERANCE) and (total_words > structural_metric_params.ZERO_TOLERANCE)):
            
            score = (structural_metric_params.FLESCH_CONSTANT_1 - structural_metric_params.FLESCH_CONSTANT_2 * (total_words / total_sentences) - structural_metric_params.FLESCH_CONSTANT_3 * (total_syllables / total_words))
            
            return max(structural_metric_params.MIN_READABILITY_SCORE, min(structural_metric_params.MAX_READABILITY_SCORE, score))
       
        return structural_metric_params.NEUTRAL_READABILITY_SCORE
    

    def _count_syllables(self, word: str) -> int:
        """
        Approximate syllable count for a word
        """
        word               = word.lower()
        vowels             = 'aeiouy'
        syllable_count     = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1

            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if (word.endswith('e')):
            syllable_count -= 1
        
        # Ensure at least one syllable
        if (syllable_count == 0):
            syllable_count = 1
        
        return syllable_count
    

    def _detect_repetitive_patterns(self, words: List[str]) -> float:
        """
        Detect repetitive patterns in text
        AI text sometimes shows more repetition
        """
        if (len(words) < structural_metric_params.MIN_WORDS_FOR_REPETITION):
            return structural_metric_params.ZERO_VALUE
        
        window_size = structural_metric_params.REPETITION_WINDOW_SIZE
        repetitions = 0
        
        for i in range(len(words) - window_size):
            window       = words[i:i + window_size]
            word_counts  = Counter(window)
            # Count words that appear more than once
            repetitions += sum(1 for count in word_counts.values() if count > 1)
        
        # Normalize
        max_repetitions  = (len(words) - window_size) * window_size
        
        if (max_repetitions > structural_metric_params.ZERO_TOLERANCE):
            repetition_score = repetitions / max_repetitions
            return min(structural_metric_params.MAX_PROBABILITY, repetition_score)
        
        return structural_metric_params.ZERO_VALUE

    
    def _calculate_ngram_diversity(self, words: List[str], n: int = 2) -> float:
        """
        Calculate n-gram diversity: Higher diversity often indicates human writing
        """
        if (len(words) < structural_metric_params.MIN_WORDS_FOR_NGRAM):
            return structural_metric_params.ZERO_VALUE
        
        # Generate n-grams
        ngrams        = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        total_ngrams  = len(ngrams)
        
        if total_ngrams > structural_metric_params.ZERO_TOLERANCE:
            unique_ngrams = len(set(ngrams))
            diversity     = unique_ngrams / total_ngrams
            return min(structural_metric_params.MAX_PROBABILITY, diversity)
        
        return structural_metric_params.ZERO_VALUE
    

    def _calculate_synthetic_probability(self, features: Dict[str, Any]) -> tuple:
        """
        Calculate synthetic probability based on structural features: Returns raw score and confidence
        """
        synthetic_indicators = list()
        params               = structural_metric_params
        
        # Low burstiness suggests synthetic (AI is more consistent)
        if (features['burstiness_score'] < params.BURSTINESS_LOW_THRESHOLD):
            synthetic_indicators.append(params.STRONG_SYNTHETIC_WEIGHT)

        elif (features['burstiness_score'] < params.BURSTINESS_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.WEAK_SYNTHETIC_WEIGHT)
        
        # High length uniformity suggests synthetic
        if (features['length_uniformity'] > params.LENGTH_UNIFORMITY_HIGH_THRESHOLD):
            synthetic_indicators.append(params.STRONG_SYNTHETIC_WEIGHT)
        
        elif (features['length_uniformity'] > params.LENGTH_UNIFORMITY_MEDIUM_THRESH):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.WEAK_SYNTHETIC_WEIGHT)
        
        # Low n-gram diversity suggests synthetic
        if (features['bigram_diversity'] < params.BIGRAM_DIVERSITY_LOW_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.VERY_WEAK_SYNTHETIC_WEIGHT)
        
        # Moderate readability suggests synthetic (AI often produces "perfect" readability)
        if (params.READABILITY_SYNTHETIC_MIN <= features['readability_score'] <= params.READABILITY_SYNTHETIC_MAX):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.VERY_WEAK_SYNTHETIC_WEIGHT)
        
        # Low repetition suggests synthetic (AI avoids excessive repetition)
        if (features['repetition_score'] < params.REPETITION_LOW_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        elif (features['repetition_score'] < params.REPETITION_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.NEUTRAL_WEIGHT)
        
        else:
            synthetic_indicators.append(params.WEAK_SYNTHETIC_WEIGHT)
        
        # Calculate raw score and confidence
        if synthetic_indicators:
            raw_score  = np.mean(synthetic_indicators)
            confidence = params.MAX_PROBABILITY - min(params.MAX_PROBABILITY, np.std(synthetic_indicators) / params.CONFIDENCE_STD_NORMALIZER)
            confidence = max(params.MIN_CONFIDENCE, min(params.MAX_CONFIDENCE, confidence))
        
        else:
            raw_score  = params.NEUTRAL_PROBABILITY
            confidence = params.NEUTRAL_CONFIDENCE
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content based on structural patterns
        """
        mixed_indicators = list()
        params           = structural_metric_params
        
        # High burstiness suggests hybrid content (inconsistent patterns)
        if (features['burstiness_score'] > params.BURSTINESS_HIGH_THRESHOLD):
            mixed_indicators.append(params.MODERATE_HYBRID_WEIGHT)
        
        # Inconsistent sentence lengths might indicate mixing
        if (features['avg_sentence_length'] > params.ZERO_TOLERANCE and features['std_sentence_length'] > features['avg_sentence_length'] * params.SENTENCE_LENGTH_VARIANCE_RATIO):
            mixed_indicators.append(params.WEAK_HYBRID_WEIGHT)
        
        # Extreme values in multiple features might indicate mixing
        extreme_features = 0
        if (features['type_token_ratio'] < params.TYPE_TOKEN_RATIO_EXTREME_LOW) or (features['type_token_ratio'] > params.TYPE_TOKEN_RATIO_EXTREME_HIGH):
            extreme_features += 1
        
        if (features['readability_score'] < params.READABILITY_EXTREME_LOW) or (features['readability_score'] > params.READABILITY_EXTREME_HIGH):
            extreme_features += 1
        
        if (extreme_features >= 2):
            mixed_indicators.append(params.WEAK_HYBRID_WEIGHT)
        
        if mixed_indicators:
            hybrid_prob = np.mean(mixed_indicators)
            return min(params.MAX_HYBRID_PROBABILITY, hybrid_prob)
        
        return params.MIN_PROBABILITY



# Export
__all__ = ["StructuralMetric"]