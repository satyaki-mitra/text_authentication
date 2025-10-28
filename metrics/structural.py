# DEPENDENCIES
import re
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from collections import Counter
from metrics.base_metric import MetricResult
from metrics.base_metric import StatisticalMetric 
from config.threshold_config import Domain
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
            { MetricResult } : MetricResult with AI/Human probabilities
        """
        try:
            # Get domain-specific thresholds
            domain                          = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds               = get_threshold_for_domain(domain)
            structural_thresholds           = domain_thresholds.structural
            
            # Extract all structural features
            features                        = self._extract_features(text)
            
            # Calculate raw AI probability based on features
            raw_ai_prob, confidence         = self._calculate_ai_probability(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob = self._apply_domain_thresholds(raw_ai_prob, structural_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                     *= structural_thresholds.confidence_multiplier
            confidence                      = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : structural_thresholds.ai_threshold,
                                                     'human_threshold' : structural_thresholds.human_threshold,
                                                     'raw_score'       : raw_ai_prob,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in {self.name} computation: {repr(e)}")
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
        ai_threshold    = thresholds.ai_threshold    # Domain-specific
        human_threshold = thresholds.human_threshold # Domain-specific
        
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
            range_width = ai_threshold - human_threshold
            
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
        
        # Calculate mixed probability based on statistical patterns
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob
        
        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob

    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract all structural features from text
        """
        # Basic tokenization
        sentences           = self._split_sentences(text)
        words               = self._tokenize_words(text)
        
        # Sentence-level features
        sentence_lengths    = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        std_sentence_length = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Word-level features
        word_lengths        = [len(w) for w in words]
        avg_word_length     = np.mean(word_lengths) if word_lengths else 0
        std_word_length     = np.std(word_lengths) if len(word_lengths) > 1 else 0
        
        # Vocabulary richness
        vocabulary_size     = len(set(words))
        type_token_ratio    = vocabulary_size / len(words) if words else 0
        
        # Punctuation analysis
        punctuation_density = self._calculate_punctuation_density(text)
        comma_frequency     = text.count(',') / len(words) if words else 0
        
        # Burstiness (variation in patterns)
        burstiness          = self._calculate_burstiness(sentence_lengths)
        
        # Uniformity scores
        length_uniformity   = 1.0 - (std_sentence_length / avg_sentence_length) if avg_sentence_length > 0 else 0
        length_uniformity   = max(0, min(1, length_uniformity))
        
        # Readability approximation (simplified)
        readability         = self._calculate_readability(text, sentences, words)
        
        # Pattern detection
        repetition_score    = self._detect_repetitive_patterns(words)
        
        # N-gram analysis
        bigram_diversity    = self._calculate_ngram_diversity(words, n = 2)
        trigram_diversity   = self._calculate_ngram_diversity(words, n = 3)
        
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
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    

    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words
        """
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        return words
    

    def _calculate_punctuation_density(self, text: str) -> float:
        """
        Calculate punctuation density
        """
        punctuation = re.findall(r'[^\w\s]', text)
        total_chars = len(text)
        
        return len(punctuation) / total_chars if total_chars > 0 else 0
    

    def _calculate_burstiness(self, values: List[float]) -> float:
        """
        Calculate burstiness score (variation in patterns)
        Higher burstiness typically indicates human writing
        """
        if (len(values) < 2):
            return 0.0
        
        mean_val   = np.mean(values)
        std_val    = np.std(values)
        
        if (mean_val == 0):
            return 0.0
        
        # Coefficient of variation
        cv         = std_val / mean_val
        
        # Normalize to 0-1 range
        burstiness = min(1.0, cv / 2.0)
        
        return burstiness
    

    def _calculate_readability(self, text: str, sentences: List[str], words: List[str]) -> float:
        """
        Calculate simplified readability score
        (Approximation of Flesch Reading Ease)
        """
        if not sentences or not words:
            return 0.0
        
        total_sentences = len(sentences)
        total_words     = len(words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease approximation
        if ((total_sentences > 0) and (total_words > 0)):
            score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
            return max(0, min(100, score))
       
        # Neutral score
        return 50.0 
    

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
        if (len(words) < 10):
            return 0.0
        
        # Check for repeated words in close proximity
        window_size = 10
        repetitions = 0
        
        for i in range(len(words) - window_size):
            window       = words[i:i + window_size]
            word_counts  = Counter(window)
            # Count words that appear more than once
            repetitions += sum(1 for count in word_counts.values() if count > 1)
        
        # Normalize
        max_repetitions  = (len(words) - window_size) * window_size
        repetition_score = repetitions / max_repetitions if max_repetitions > 0 else 0
        
        return repetition_score

    
    def _calculate_ngram_diversity(self, words: List[str], n: int = 2) -> float:
        """
        Calculate n-gram diversity
        Higher diversity often indicates human writing
        """
        if (len(words) < n):
            return 0.0
        
        # Generate n-grams
        ngrams        = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        # Calculate diversity as ratio of unique n-grams to total n-grams
        unique_ngrams = len(set(ngrams))
        total_ngrams  = len(ngrams)
        
        diversity     = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
        
        return diversity
    

    def _calculate_ai_probability(self, features: Dict[str, Any]) -> tuple:
        """
        Calculate AI probability based on structural features
        Returns raw score and confidence
        """
        ai_indicators = list()
        
        # Low burstiness suggests AI (AI is more consistent)
        if (features['burstiness_score'] < 0.3):
            # Strong AI indicator
            ai_indicators.append(0.7)

        elif (features['burstiness_score'] < 0.5):
            # Moderate AI indicator
            ai_indicators.append(0.5)

        else:
            # Weak AI indicator
            ai_indicators.append(0.3)
        
        # High length uniformity suggests AI
        if (features['length_uniformity'] > 0.7):
            # Strong AI indicator
            ai_indicators.append(0.7)

        elif (features['length_uniformity'] > 0.5):
            # Moderate AI indicator
            ai_indicators.append(0.5)

        else:
            # Weak AI indicator
            ai_indicators.append(0.3)
        
        # Low n-gram diversity suggests AI
        if (features['bigram_diversity'] < 0.7):
            # Moderate AI indicator
            ai_indicators.append(0.6)

        else:
            # Weak AI indicator
            ai_indicators.append(0.4)
        
        # Moderate readability suggests AI (AI often produces "perfect" readability)
        if (60 <= features['readability_score'] <= 75):
            # Moderate AI indicator
            ai_indicators.append(0.6)

        else:
            # Weak AI indicator
            ai_indicators.append(0.4)
        
        # Low repetition suggests AI (AI avoids excessive repetition)
        if (features['repetition_score'] < 0.1):
            # Moderate AI indicator
            ai_indicators.append(0.6)

        elif (features['repetition_score'] < 0.2):
            # Neutral
            ai_indicators.append(0.5)

        else:
            # Weak AI indicator
            ai_indicators.append(0.3)
        
        # Calculate raw score and confidence
        raw_score  = np.mean(ai_indicators) if ai_indicators else 0.5
        confidence = 1.0 - (np.std(ai_indicators) / 0.5) if ai_indicators else 0.5
        confidence = max(0.1, min(0.9, confidence))
        
        return raw_score, confidence
    

    def _calculate_mixed_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of mixed AI/Human content based on structural patterns
        """
        mixed_indicators = []
        
        # High burstiness suggests mixed content (inconsistent patterns)
        if features['burstiness_score'] > 0.6:
            mixed_indicators.append(0.4)
        
        # Inconsistent sentence lengths might indicate mixing
        if (features['std_sentence_length'] > features['avg_sentence_length'] * 0.8):
            mixed_indicators.append(0.3)
        
        # Extreme values in multiple features might indicate mixing
        extreme_features = 0
        if (features['type_token_ratio'] < 0.3) or (features['type_token_ratio'] > 0.9):
            extreme_features += 1
        if (features['readability_score'] < 20) or (features['readability_score'] > 90):
            extreme_features += 1
        
        if (extreme_features >= 2):
            mixed_indicators.append(0.3)
        
        return min(0.3, np.mean(mixed_indicators)) if mixed_indicators else 0.0


# Export
__all__ = ["StructuralMetric"]