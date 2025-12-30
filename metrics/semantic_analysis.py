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
from metrics.base_metric import BaseMetric
from models.model_manager import get_model_manager
from config.constants import semantic_analysis_params
from sklearn.metrics.pairwise import cosine_similarity
from config.threshold_config import get_threshold_for_domain


class SemanticAnalysisMetric(BaseMetric):
    """
    Semantic coherence and consistency analysis
    
    Measures (Aligned with Documentation):
    - Semantic similarity between sentences
    - Topic consistency across text
    - Coherence and logical flow
    - Repetition patterns and redundancy
    - Contextual consistency
    """
    def __init__(self):
        super().__init__(name        = "semantic_analysis",
                         description = "Semantic coherence, repetition patterns, and contextual consistency analysis",
                        )

        self.sentence_model = None
    

    def initialize(self) -> bool:
        """
        Initialize the semantic analysis metric
        """
        try:
            logger.info("Initializing semantic analysis metric...")
            
            # Load sentence transformer for semantic embeddings
            model_manager       = get_model_manager()
            self.sentence_model = model_manager.load_model("semantic_primary")
            
            self.is_initialized = True

            logger.success("Semantic analysis metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic analysis metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute semantic analysis measures with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            params = semantic_analysis_params
            
            if (not text or (len(text.strip()) < params.MIN_TEXT_LENGTH_FOR_ANALYSIS)):
                return self._default_result(error = "Text too short for semantic analysis")
            
            # Get domain-specific thresholds
            domain                                      = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                           = get_threshold_for_domain(domain)
            semantic_thresholds                         = domain_thresholds.semantic
            
            # Calculate comprehensive semantic features
            features                                    = self._calculate_semantic_features(text)
            
            # Calculate raw semantic score (0-1 scale)
            raw_semantic_score, confidence              = self._analyze_semantic_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob = self._apply_domain_thresholds(raw_score  = raw_semantic_score, 
                                                                                        thresholds = semantic_thresholds, 
                                                                                        features   = features,
                                                                                       )
            
            # Apply confidence multiplier from domain thresholds
            confidence                                 *= semantic_thresholds.confidence_multiplier
            confidence                                  = max(params.MIN_CONFIDENCE, min(params.MAX_CONFIDENCE, confidence))
            
            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'          : domain.value,
                                                         'synthetic_threshold'  : semantic_thresholds.synthetic_threshold,
                                                         'authentic_threshold'  : semantic_thresholds.authentic_threshold,
                                                         'raw_score'            : raw_semantic_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis computation: {repr(e)}")
            return self._default_result(error = str(e))
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        params              = semantic_analysis_params
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
        
        # Calculate hybrid probability based on semantic variance
        hybrid_prob    = self._calculate_hybrid_probability(features = features)
        
        # Normalize to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob
        
        if (total > params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob
    

    def _calculate_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive semantic analysis features
        """
        params    = semantic_analysis_params
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if (len(sentences) < params.MIN_SENTENCES_FOR_ANALYSIS):
            return self._get_default_features()
        
        # Calculate semantic embeddings for all sentences
        sentence_embeddings, valid_sentences = self._get_sentence_embeddings(sentences = sentences)
        
        if sentence_embeddings is None:
            return self._get_default_features()
        
        # Calculate semantic similarity matrix
        similarity_matrix      = cosine_similarity(sentence_embeddings)
        
        # Calculate various semantic metrics
        coherence_score        = self._calculate_coherence(similarity_matrix = similarity_matrix)
        consistency_score      = self._calculate_consistency(similarity_matrix = similarity_matrix)
        repetition_score       = self._detect_repetition_patterns(sentences         = valid_sentences, 
                                                                  similarity_matrix = similarity_matrix,
                                                                 )

        topic_drift_score      = self._calculate_topic_drift(similarity_matrix = similarity_matrix)
        contextual_consistency = self._calculate_contextual_consistency(sentences = sentences)
        
        # Chunk-based analysis for whole-text understanding
        chunk_coherence        = self._calculate_chunk_coherence(text       = text, 
                                                                 chunk_size = params.CHUNK_SIZE_WORDS,
                                                                )
        
        return {"coherence_score"        : round(coherence_score, 4),
                "consistency_score"      : round(consistency_score, 4),
                "repetition_score"       : round(repetition_score, 4),
                "topic_drift_score"      : round(topic_drift_score, 4),
                "contextual_consistency" : round(contextual_consistency, 4),
                "avg_chunk_coherence"    : round(np.mean(chunk_coherence) if chunk_coherence else params.DEFAULT_COHERENCE, 4),
                "coherence_variance"     : round(np.var(chunk_coherence) if chunk_coherence else params.DEFAULT_COHERENCE_VARIANCE, 4),
                "num_sentences"          : len(valid_sentences),
                "num_chunks_analyzed"    : len(chunk_coherence),
               }
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        sentences = re.split(semantic_analysis_params.SENTENCE_SPLIT_PATTERN, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > semantic_analysis_params.MIN_SENTENCE_LENGTH]
    

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get semantic embeddings for sentences
        """
        try:
            if not self.sentence_model:
                return None
            
            # Filter out very short sentences that might cause issues
            valid_sentences = [s for s in sentences if len(s.strip()) > semantic_analysis_params.MIN_VALID_SENTENCE_LENGTH]
            if not valid_sentences:
                return None, None
            
            # Encode sentences to get embeddings
            embeddings = self.sentence_model.encode(valid_sentences)
            
            # Check if embeddings are valid
            if ((embeddings is None) or (len(embeddings) == 0)):
                return None, None
                
            return embeddings, valid_sentences
            
        except Exception as e:
            logger.warning(f"Sentence embedding failed: {repr(e)}")
            return None, None
    

    def _calculate_coherence(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate overall text coherence : Higher coherence = more logically connected sentences
        """
        params = semantic_analysis_params
        
        if (similarity_matrix.size == 0):
            return params.MIN_PROBABILITY
        
        # Calculate average similarity between adjacent sentences
        adjacent_similarities = list()

        for i in range(len(similarity_matrix) - 1):
            adjacent_similarities.append(similarity_matrix[i, i + 1])
        
        if (not adjacent_similarities):
            return params.MIN_PROBABILITY
        
        return np.mean(adjacent_similarities)
    

    def _calculate_consistency(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate topic consistency throughout the text : Lower variance in similarities = more consistent
        """
        params = semantic_analysis_params
        
        if (similarity_matrix.size == 0):
            return params.MIN_PROBABILITY
        
        # Calculate variance of similarities (lower variance = more consistent)
        all_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        if (len(all_similarities) == 0):
            return params.MIN_PROBABILITY
        
        variance    = np.var(all_similarities)
        # Convert to consistency score (higher = more consistent)
        consistency = params.MAX_PROBABILITY - min(params.MAX_PROBABILITY, variance * params.SIMILARITY_VARIANCE_FACTOR)

        return max(params.MIN_PROBABILITY, consistency)
    

    def _detect_repetition_patterns(self, sentences: List[str], similarity_matrix: np.ndarray) -> float:
        """
        Detect repetition patterns in semantic content : AI text sometimes shows more semantic repetition
        """
        params = semantic_analysis_params
        
        if (len(sentences) < params.MIN_SENTENCES_FOR_REPETITION):
            return params.MIN_PROBABILITY
        
        # Look for high similarity between non-adjacent sentences
        repetition_count  = 0
        total_comparisons = 0
        
        for i in range(len(sentences)):
            for j in range(i + 2, len(sentences)):  # Skip adjacent sentences
                # High semantic similarity
                if (similarity_matrix[i, j] > params.REPETITION_SIMILARITY_THRESHOLD):  
                    repetition_count += 1
                
                total_comparisons += 1
        
        if (total_comparisons == 0):
            return params.MIN_PROBABILITY
        
        repetition_score = repetition_count / total_comparisons
        
        # Scale to make differences more noticeable
        return min(params.MAX_PROBABILITY, repetition_score * params.REPETITION_SCORE_SCALING)
    

    def _calculate_topic_drift(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate topic drift throughout the text : Higher drift = less focused content
        """
        params = semantic_analysis_params
        
        if (len(similarity_matrix) < 3):
            return params.MIN_PROBABILITY
        
        # Calculate similarity between beginning and end sections
        start_size         = min(params.START_SECTION_SIZE, len(similarity_matrix) // params.SECTION_SIZE_RATIO)
        end_size           = min(params.END_SECTION_SIZE, len(similarity_matrix) // params.SECTION_SIZE_RATIO)
        
        start_indices      = list(range(start_size))
        end_indices        = list(range(len(similarity_matrix) - end_size, len(similarity_matrix)))
        
        cross_similarities = list()

        for i in start_indices:
            for j in end_indices:
                cross_similarities.append(similarity_matrix[i, j])
        
        if not cross_similarities:
            return params.MIN_PROBABILITY
        
        avg_cross_similarity = np.mean(cross_similarities)
        # Lower similarity between start and end = higher topic drift
        topic_drift          = params.MAX_PROBABILITY - avg_cross_similarity

        return max(params.MIN_PROBABILITY, topic_drift)
    

    def _calculate_contextual_consistency(self, sentences: List[str]) -> float:
        """
        Calculate contextual consistency using keyword and entity analysis
        """
        params = semantic_analysis_params
        
        if (len(sentences) < params.MIN_SENTENCES_FOR_ANALYSIS):
            return params.MIN_PROBABILITY
        
        # Simple keyword consistency analysis : Extract meaningful words (nouns, adjectives)
        all_words = list()

        for sentence in sentences:
            words = re.findall(params.WORD_EXTRACTION_PATTERN, sentence.lower())
            all_words.extend(words)
        
        if (len(all_words) < params.MIN_WORDS_FOR_KEYWORD_ANALYSIS):
            return params.MIN_PROBABILITY
        
        # Calculate how consistently keywords are used across sentences
        word_freq    = Counter(all_words)
        top_keywords = [word for word, count in word_freq.most_common(params.TOP_KEYWORDS_COUNT) if count > params.MIN_KEYWORD_FREQUENCY]
        
        if not top_keywords:
            return params.MIN_PROBABILITY
        
        # Check if top keywords appear consistently across sentences
        keyword_presence = list()

        for keyword in top_keywords:
            sentences_with_keyword = sum(1 for sentence in sentences if keyword in sentence.lower())
            presence_ratio         = sentences_with_keyword / len(sentences)
            keyword_presence.append(presence_ratio)
        
        consistency = np.mean(keyword_presence)

        return consistency
    

    def _calculate_chunk_coherence(self, text: str, chunk_size: int = 200) -> List[float]:
        """
        Calculate coherence across text chunks for whole-text analysis
        """
        params = semantic_analysis_params
        chunks = list()
        words  = text.split()
        
        # Create overlapping chunks
        overlap = int(chunk_size * params.CHUNK_OVERLAP_RATIO)
        
        for i in range(0, len(words), overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > params.MIN_CHUNK_LENGTH):  
                chunk_sentences = self._split_sentences(chunk)
                
                if (len(chunk_sentences) >= params.MIN_SENTENCES_PER_CHUNK):
                    sentence_embeddings, valid_sentences = self._get_sentence_embeddings(sentences = chunk_sentences)
                    
                    if ((sentence_embeddings is not None) and (len(sentence_embeddings) >= params.MIN_SENTENCES_PER_CHUNK)):
                        similarity_matrix = cosine_similarity(sentence_embeddings)
                        coherence         = self._calculate_coherence(similarity_matrix)
                        chunks.append(coherence)
        
        return chunks if chunks else [params.DEFAULT_COHERENCE]
    

    def _analyze_semantic_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze semantic patterns to determine RAW semantic score (0-1 scale)
        """
        params = semantic_analysis_params
        
        # Check feature validity first
        required_features = ['coherence_score', 'consistency_score', 'repetition_score', 'topic_drift_score', 'coherence_variance']
    
        valid_features    = [features.get(feat, params.MIN_PROBABILITY) for feat in required_features if features.get(feat, params.MIN_PROBABILITY) > params.ZERO_TOLERANCE]
    
        if (len(valid_features) < params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return params.NEUTRAL_PROBABILITY, params.LOW_FEATURE_CONFIDENCE
        
        # Initialize synthetic indicator list
        synthetic_indicators = list()

        # AI text often has very high coherence (too perfect)
        if (features['coherence_score'] > params.COHERENCE_HIGH_THRESHOLD):
            # Suspiciously high coherence
            synthetic_indicators.append(params.STRONG_SYNTHETIC_WEIGHT)
       
        elif (features['coherence_score'] > params.COHERENCE_MEDIUM_THRESHOLD):
            # Moderate coherence
            synthetic_indicators.append(params.MEDIUM_SYNTHETIC_WEIGHT)
       
        else:
            # Low coherence - more human-like
            synthetic_indicators.append(params.LOW_SYNTHETIC_WEIGHT)
        
        # Very high consistency suggests AI (unnaturally consistent)
        if (features['consistency_score'] > params.CONSISTENCY_HIGH_THRESHOLD):
            synthetic_indicators.append(params.STRONG_SYNTHETIC_WEIGHT)
        
        elif (features['consistency_score'] > params.CONSISTENCY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.VERY_LOW_SYNTHETIC_WEIGHT)
        
        # High repetition suggests AI
        if (features['repetition_score'] > params.REPETITION_HIGH_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        elif (features['repetition_score'] > params.REPETITION_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.VERY_WEAK_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.LOW_SYNTHETIC_WEIGHT)
        
        # Very low topic drift suggests AI (stays too focused)
        if (features['topic_drift_score'] < params.TOPIC_DRIFT_LOW_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        elif (features['topic_drift_score'] < params.TOPIC_DRIFT_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.WEAK_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.VERY_LOW_SYNTHETIC_WEIGHT)
        
        # Low coherence variance across chunks suggests AI
        if (features['coherence_variance'] < params.COHERENCE_VARIANCE_LOW_THRESHOLD):
            synthetic_indicators.append(params.MODERATE_SYNTHETIC_WEIGHT)
        
        elif (features['coherence_variance'] < params.COHERENCE_VARIANCE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(params.VERY_WEAK_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(params.LOW_SYNTHETIC_WEIGHT)
        
        # Calculate raw score and confidence
        if synthetic_indicators:
            raw_score  = np.mean(synthetic_indicators)
            confidence = params.MAX_PROBABILITY - (np.std(synthetic_indicators) / params.CONFIDENCE_STD_NORMALIZER)
            confidence = max(params.MIN_CONFIDENCE, min(params.MAX_CONFIDENCE, confidence))
        
        else:
            raw_score  = params.NEUTRAL_PROBABILITY
            confidence = params.NEUTRAL_CONFIDENCE
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content
        """
        mixed_indicators = list()
        params           = semantic_analysis_params
        
        # Moderate coherence values might indicate mixing
        if (params.COHERENCE_MIXED_MIN <= features['coherence_score'] <= params.COHERENCE_MIXED_MAX):
            mixed_indicators.append(params.WEAK_HYBRID_WEIGHT)
        
        else:
            mixed_indicators.append(params.MIN_PROBABILITY)
        
        # High coherence variance suggests mixed content
        if (features['coherence_variance'] > params.COHERENCE_VARIANCE_HIGH_THRESHOLD):
            mixed_indicators.append(params.MODERATE_HYBRID_WEIGHT)
        
        elif (features['coherence_variance'] > params.COHERENCE_VARIANCE_MEDIUM_THRESHOLD):
            mixed_indicators.append(params.WEAK_HYBRID_WEIGHT)
        
        else:
            mixed_indicators.append(params.MIN_PROBABILITY)
        
        # Inconsistent repetition patterns
        if (params.REPETITION_MIXED_MIN <= features['repetition_score'] <= params.REPETITION_MIXED_MAX):
            mixed_indicators.append(params.WEAK_HYBRID_WEIGHT)
        
        else:
            mixed_indicators.append(params.MIN_PROBABILITY)
        
        if mixed_indicators:
            hybrid_prob = np.mean(mixed_indicators)
            return min(params.MAX_HYBRID_PROBABILITY, hybrid_prob)
        
        return params.MIN_PROBABILITY
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        params = semantic_analysis_params
        
        return {"coherence_score"        : params.DEFAULT_COHERENCE,
                "consistency_score"      : params.DEFAULT_CONSISTENCY,
                "repetition_score"       : params.DEFAULT_REPETITION,
                "topic_drift_score"      : params.DEFAULT_TOPIC_DRIFT,
                "contextual_consistency" : params.DEFAULT_CONTEXTUAL_CONSISTENCY,
                "avg_chunk_coherence"    : params.DEFAULT_CHUNK_COHERENCE,
                "coherence_variance"     : params.DEFAULT_COHERENCE_VARIANCE,
                "num_sentences"          : 0,
                "num_chunks_analyzed"    : 0,
               }
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.sentence_model = None
        super().cleanup()




# Export
__all__ = ["SemanticAnalysisMetric"]