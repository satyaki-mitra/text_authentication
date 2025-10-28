# DEPENDENCIES
import math
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from collections import Counter
from metrics.base_metric import BaseMetric
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from models.model_manager import get_model_manager
from config.threshold_config import get_threshold_for_domain


class EntropyMetric(BaseMetric):
    """
    Enhanced entropy analysis for text randomness and predictability
    
    Measures (Aligned with Documentation):
    - Character-level entropy and diversity
    - Word-level entropy and burstiness  
    - Token-level diversity and unpredictability in sequences
    - Entropy distribution across text chunks
    - AI-specific pattern detection
    """
    def __init__(self):
        super().__init__(name        = "entropy",
                         description = "Token-level diversity and unpredictability in text sequences",
                        )
        self.tokenizer = None
    

    def initialize(self) -> bool:
        """
        Initialize the entropy metric
        """
        try:
            logger.info("Initializing entropy metric...")
            
            # Load tokenizer for token-level analysis
            model_manager = get_model_manager()
            gpt_model     = model_manager.load_model("perplexity_gpt2")
            
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
        Compute enhanced entropy measures for text with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if (not text or (len(text.strip()) < 50)):
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for entropy analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                          = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds               = get_threshold_for_domain(domain)
            entropy_thresholds              = domain_thresholds.entropy
            
            # Calculate comprehensive entropy features
            features                        = self._calculate_enhanced_entropy_features(text)
            
            # Calculate raw entropy score (0-1 scale)
            raw_entropy_score, confidence   = self._analyze_entropy_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob = self._apply_domain_thresholds(raw_entropy_score, entropy_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                     *= entropy_thresholds.confidence_multiplier
            confidence                      = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : entropy_thresholds.ai_threshold,
                                                     'human_threshold' : entropy_thresholds.human_threshold,
                                                     'raw_score'       : raw_entropy_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in entropy computation: {repr(e)}")
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
        ai_threshold    = thresholds.ai_threshold    # e.g., 0.55 for GENERAL, 0.50 for ACADEMIC
        human_threshold = thresholds.human_threshold # e.g., 0.45 for GENERAL, 0.40 for ACADEMIC
        
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
        
        # Calculate mixed probability based on entropy variance
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob

        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob
    

    def _calculate_enhanced_entropy_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive entropy measures including document-required features
        """
        # Basic entropy measures
        char_entropy              = self._calculate_character_entropy(text)
        word_entropy              = self._calculate_word_entropy(text)
        token_entropy             = self._calculate_token_entropy(text) if self.tokenizer else 0.0
        
        # DOCUMENT-REQUIRED: Token-level diversity
        token_diversity           = self._calculate_token_diversity(text)
        
        # DOCUMENT-REQUIRED: Unpredictability in sequences
        sequence_unpredictability = self._calculate_sequence_unpredictability(text)
        
        # Chunk-based analysis for whole-text understanding
        chunk_entropies           = self._calculate_chunk_entropy(text, chunk_size=100)
        entropy_variance          = np.var(chunk_entropies) if chunk_entropies else 0.0
        avg_chunk_entropy         = np.mean(chunk_entropies) if chunk_entropies else 0.0
        
        # AI-specific pattern detection
        ai_pattern_score          = self._detect_ai_entropy_patterns(text)
        
        # Predictability measures
        predictability            = 1.0 - min(1.0, char_entropy / 4.0)
        
        return {"char_entropy"              : round(char_entropy, 4),
                "word_entropy"              : round(word_entropy, 4),
                "token_entropy"             : round(token_entropy, 4),
                "token_diversity"           : round(token_diversity, 4), 
                "sequence_unpredictability" : round(sequence_unpredictability, 4),  
                "entropy_variance"          : round(entropy_variance, 4),
                "avg_chunk_entropy"         : round(avg_chunk_entropy, 4),
                "predictability_score"      : round(predictability, 4),
                "ai_pattern_score"          : round(ai_pattern_score, 4),
                "num_chunks_analyzed"       : len(chunk_entropies),
               }
    

    def _calculate_character_entropy(self, text: str) -> float:
        """
        Calculate character-level entropy
        """
        # Clean text and convert to lowercase
        clean_text = ''.join(c for c in text.lower() if c.isalnum() or c.isspace())
        
        if not clean_text:
            return 0.0
        
        # Count character frequencies
        char_counts = Counter(clean_text)
        total_chars = len(clean_text)
        
        # Calculate entropy
        entropy    = 0.0

        for count in char_counts.values():
            probability = count / total_chars
            entropy    -= probability * math.log2(probability)
        
        return entropy
    

    def _calculate_word_entropy(self, text: str) -> float:
        """
        Calculate word-level entropy
        """
        words = text.lower().split()
        if (len(words) < 5):
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy     = 0.0

        for count in word_counts.values():
            probability = count / total_words
            entropy    -= probability * math.log2(probability)
        
        return entropy
    

    def _calculate_token_entropy(self, text: str) -> float:
        """
        Calculate token-level entropy using GPT-2 tokenizer
        """
        try:
            if not self.tokenizer:
                return 0.0
            
            # Length check before tokenization
            if (len(text.strip()) < 10):
                return 0.0

            # Tokenize text
            tokens = self.tokenizer.encode(text, 
                                           add_special_tokens = False, 
                                           truncation         = True, 
                                          )
            
            if (len(tokens) < 10):
                return 0.0
            
            token_counts = Counter(tokens)
            total_tokens = len(tokens)
            
            entropy      = 0.0

            for count in token_counts.values():
                probability = count / total_tokens
                entropy    -= probability * math.log2(probability)
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Token entropy calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_token_diversity(self, text: str) -> float:
        """
        Calculate token-level diversity : Higher diversity = more human-like
        """
        if not self.tokenizer:
            return 0.0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if (len(tokens) < 10):
                return 0.0
            
            unique_tokens = len(set(tokens))
            total_tokens  = len(tokens)
            
            # Type-token ratio for tokens
            diversity     = unique_tokens / total_tokens

            return diversity
            
        except Exception as e:
            logger.warning(f"Token diversity calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_sequence_unpredictability(self, text: str) -> float:
        """
        Calculate unpredictability in text sequences, it measures how unpredictable the token sequences are
        """
        if not self.tokenizer:
            return 0.0
        
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if (len(tokens) < 20):
                return 0.0
            
            # Calculate bigram unpredictability
            bigrams          = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            bigram_counts    = Counter(bigrams)
            total_bigrams    = len(bigrams)
            
            # Higher entropy = more unpredictable sequences
            sequence_entropy = 0.0

            for count in bigram_counts.values():
                probability       = count / total_bigrams
                sequence_entropy -= probability * math.log2(probability)
            
            # Normalize to 0-1 scale : Assuming max ~8 bits
            normalized_entropy = min(1.0, sequence_entropy / 8.0)  
            
            return normalized_entropy
            
        except Exception as e:
            logger.warning(f"Sequence unpredictability calculation failed: {repr(e)}")
            return 0.0
    

    def _calculate_chunk_entropy(self, text: str, chunk_size: int = 100) -> List[float]:
        """
        Calculate entropy distribution across text chunks
        """
        chunks = list()
        words  = text.split()
        
        # Create overlapping chunks for better analysis
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > 20):  
                entropy = self._calculate_character_entropy(chunk)
                chunks.append(entropy)
        
        return chunks
    

    def _detect_ai_entropy_patterns(self, text: str) -> float:
        """
        Detect AI-specific entropy patterns: AI text often shows specific entropy signatures
        """
        patterns_detected = 0
        total_patterns    = 4
        
        # Overly consistent character distribution
        char_entropy      = self._calculate_character_entropy(text)
        
        # AI tends to be more consistent
        if (char_entropy < 3.8):  
            patterns_detected += 1
        
        # Low token diversity
        token_diversity = self._calculate_token_diversity(text)

        # AI reuses tokens more
        if (token_diversity < 0.7):  
            patterns_detected += 1
        
        # Predictable sequences
        sequence_unpredictability = self._calculate_sequence_unpredictability(text)
        
        # AI sequences are more predictable
        if (sequence_unpredictability < 0.4):  
            patterns_detected += 1
        
        # Low entropy variance across chunks
        chunk_entropies  = self._calculate_chunk_entropy(text, chunk_size = 100)
        entropy_variance = np.var(chunk_entropies) if chunk_entropies else 0.0
        
        # AI maintains consistent entropy
        if (entropy_variance < 0.2):  
            patterns_detected += 1
        
        return patterns_detected / total_patterns
    

    def _analyze_entropy_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze entropy patterns to determine RAW entropy score (0-1 scale)
        This raw score will later be converted using domain thresholds
        """
        # Check feature validity
        valid_features = [score for score in [features.get('char_entropy', 0),
                                              features.get('token_diversity', 0), 
                                              features.get('sequence_unpredictability', 0),
                                              features.get('ai_pattern_score', 0)
                                             ] if score > 0
                         ]
        
        if (len(valid_features) < 2):
            # Low confidence if insufficient features
            return 0.5, 0.3  

        ai_indicators = list()
        
        # AI text often has lower character entropy (more predictable)
        if (features['char_entropy'] < 3.5):
            # Strong AI indicator
            ai_indicators.append(0.8)  

        elif (features['char_entropy'] < 4.0):
            # Moderate AI indicator
            ai_indicators.append(0.6)  

        else:
            # Weak AI indicator
            ai_indicators.append(0.2)  
        
        # Low entropy variance suggests AI (consistent patterns)
        if (features['entropy_variance'] < 0.1):
            # Very strong AI indicator
            ai_indicators.append(0.9)  

        elif (features['entropy_variance'] < 0.3):
            # Neutral
            ai_indicators.append(0.5)  

        else:
            # Strong human indicator
            ai_indicators.append(0.1)  
        
        # Low token diversity suggests AI
        if (features['token_diversity'] < 0.6):
            ai_indicators.append(0.7)

        elif (features['token_diversity'] < 0.8):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # Low sequence unpredictability suggests AI
        if (features['sequence_unpredictability'] < 0.3):
            ai_indicators.append(0.8)

        elif (features['sequence_unpredictability'] < 0.5):
            ai_indicators.append(0.5)

        else:
            ai_indicators.append(0.2)
        
        # High AI pattern score suggests AI
        if (features['ai_pattern_score'] > 0.75):
            ai_indicators.append(0.9)
        
        elif (features['ai_pattern_score'] > 0.5):
            ai_indicators.append(0.7)
        
        else:
            ai_indicators.append(0.3)
        
        # Calculate raw score and confidence
        raw_score  = np.mean(ai_indicators) if ai_indicators else 0.5
        confidence = 1.0 - (np.std(ai_indicators) / 0.5) if ai_indicators else 0.5
        confidence = max(0.1, min(0.9, confidence))
        
        return raw_score, confidence
    

    def _calculate_mixed_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of mixed AI/Human content with better indicators
        """
        mixed_indicators = list()
        
        # High entropy variance suggests mixed content
        entropy_variance = features.get('entropy_variance', 0)
        
        if (entropy_variance > 0.5):
            # Strong mixed indicator
            mixed_indicators.append(0.6)  

        elif (entropy_variance > 0.3):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        # Inconsistent patterns across different entropy measures
        char_entropy = features.get('char_entropy', 0)
        word_entropy = features.get('word_entropy', 0)
        
        if ((char_entropy > 0) and (word_entropy > 0)):
            entropy_discrepancy = abs(char_entropy - word_entropy)

            # Large discrepancy suggests mixing
            if (entropy_discrepancy > 1.0):  
                mixed_indicators.append(0.4)
        
        # Moderate AI pattern score might indicate mixing
        ai_pattern_score = features.get('ai_pattern_score', 0)
        if (0.4 <= ai_pattern_score <= 0.6):
            mixed_indicators.append(0.3)
        
        mixed_probability = min(0.4, np.mean(mixed_indicators)) if mixed_indicators else 0.0
        
        return mixed_probability


    def cleanup(self):
        """
        Clean up resources
        """
        self.tokenizer = None
        super().cleanup()



# Export
__all__ = ["EntropyMetric"]
