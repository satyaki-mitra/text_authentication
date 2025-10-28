# DEPENDENCIES
import re
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from collections import Counter
from config.threshold_config import Domain
from metrics.base_metric import BaseMetric
from metrics.base_metric import MetricResult
from models.model_manager import get_model_manager
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
            if (not text or (len(text.strip()) < 50)):
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for semantic analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                          = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds               = get_threshold_for_domain(domain)
            semantic_thresholds             = domain_thresholds.semantic_analysis
            
            # Calculate comprehensive semantic features
            features                        = self._calculate_semantic_features(text)
            
            # Calculate raw semantic score (0-1 scale)
            raw_semantic_score, confidence  = self._analyze_semantic_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob = self._apply_domain_thresholds(raw_semantic_score, semantic_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                     *= semantic_thresholds.confidence_multiplier
            confidence                      = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : semantic_thresholds.ai_threshold,
                                                     'human_threshold' : semantic_thresholds.human_threshold,
                                                     'raw_score'       : raw_semantic_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis computation: {repr(e)}")
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
        ai_threshold    = thresholds.ai_threshold    # e.g., 0.65 for GENERAL, 0.70 for ACADEMIC
        human_threshold = thresholds.human_threshold # e.g., 0.35 for GENERAL, 0.30 for ACADEMIC
        
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
        
        # Calculate mixed probability based on semantic variance
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob
        
        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob
    

    def _calculate_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive semantic analysis features
        """
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if (len(sentences) < 3):
            return self._get_default_features()
        
        # Calculate semantic embeddings for all sentences
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        
        if sentence_embeddings is None:
            return self._get_default_features()
        
        # Calculate semantic similarity matrix
        similarity_matrix      = cosine_similarity(sentence_embeddings)
        
        # Calculate various semantic metrics
        coherence_score        = self._calculate_coherence(similarity_matrix)
        consistency_score      = self._calculate_consistency(similarity_matrix)
        repetition_score       = self._detect_repetition_patterns(sentences, similarity_matrix)
        topic_drift_score      = self._calculate_topic_drift(similarity_matrix)
        contextual_consistency = self._calculate_contextual_consistency(sentences)
        
        # Chunk-based analysis for whole-text understanding
        chunk_coherence        = self._calculate_chunk_coherence(text, chunk_size=200)
        
        return {"coherence_score"        : round(coherence_score, 4),
                "consistency_score"      : round(consistency_score, 4),
                "repetition_score"       : round(repetition_score, 4),
                "topic_drift_score"      : round(topic_drift_score, 4),
                "contextual_consistency" : round(contextual_consistency, 4),
                "avg_chunk_coherence"    : round(np.mean(chunk_coherence) if chunk_coherence else 0.0, 4),
                "coherence_variance"     : round(np.var(chunk_coherence) if chunk_coherence else 0.0, 4),
                "num_sentences"          : len(sentences),
                "num_chunks_analyzed"    : len(chunk_coherence),
               }
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get semantic embeddings for sentences
        """
        try:
            if not self.sentence_model:
                return None
            
            # Filter out very short sentences that might cause issues
            valid_sentences = [s for s in sentences if len(s.strip()) > 5]
            if not valid_sentences:
                return None
            
            # Encode sentences to get embeddings
            embeddings = self.sentence_model.encode(valid_sentences)
            
            # Check if embeddings are valid
            if ((embeddings is None) or (len(embeddings) == 0)):
                return None
                
            return embeddings
            
        except Exception as e:
            logger.warning(f"Sentence embedding failed: {repr(e)}")
            return None
    

    def _calculate_coherence(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate overall text coherence : Higher coherence = more logically connected sentences
        """
        if similarity_matrix.size == 0:
            return 0.0
        
        # Calculate average similarity between adjacent sentences
        adjacent_similarities = list()

        for i in range(len(similarity_matrix) - 1):
            adjacent_similarities.append(similarity_matrix[i, i + 1])
        
        if (not adjacent_similarities):
            return 0.0
        
        return np.mean(adjacent_similarities)
    

    def _calculate_consistency(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate topic consistency throughout the text : Lower variance in similarities = more consistent
        """
        if (similarity_matrix.size == 0):
            return 0.0
        
        # Calculate variance of similarities (lower variance = more consistent)
        all_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        if (len(all_similarities) == 0):
            return 0.0
        
        variance    = np.var(all_similarities)
        # Convert to consistency score (higher = more consistent)
        consistency = 1.0 - min(1.0, variance * 5.0)  # Normalize

        return max(0.0, consistency)
    

    def _detect_repetition_patterns(self, sentences: List[str], similarity_matrix: np.ndarray) -> float:
        """
        Detect repetition patterns in semantic content : AI text sometimes shows more semantic repetition
        """
        if (len(sentences) < 5):
            return 0.0
        
        # Look for high similarity between non-adjacent sentences
        repetition_count  = 0
        total_comparisons = 0
        
        for i in range(len(sentences)):
            for j in range(i + 2, len(sentences)):  # Skip adjacent sentences
                # High semantic similarity
                if (similarity_matrix[i, j] > 0.8):  
                    repetition_count += 1
                
                total_comparisons += 1
        
        if (total_comparisons == 0):
            return 0.0
        
        repetition_score = repetition_count / total_comparisons
        
        # Scale to make differences more noticeable
        return min(1.0, repetition_score * 3.0)  
    

    def _calculate_topic_drift(self, similarity_matrix: np.ndarray) -> float:
        """
        Calculate topic drift throughout the text : Higher drift = less focused content
        """
        if (len(similarity_matrix) < 3):
            return 0.0
        
        # Calculate similarity between beginning and end sections
        start_size         = min(3, len(similarity_matrix) // 3)
        end_size           = min(3, len(similarity_matrix) // 3)
        
        start_indices      = list(range(start_size))
        end_indices        = list(range(len(similarity_matrix) - end_size, len(similarity_matrix)))
        
        cross_similarities = list()

        for i in start_indices:
            for j in end_indices:
                cross_similarities.append(similarity_matrix[i, j])
        
        if not cross_similarities:
            return 0.0
        
        avg_cross_similarity = np.mean(cross_similarities)
        # Lower similarity between start and end = higher topic drift
        topic_drift          = 1.0 - avg_cross_similarity

        return max(0.0, topic_drift)
    

    def _calculate_contextual_consistency(self, sentences: List[str]) -> float:
        """
        Calculate contextual consistency using keyword and entity analysis
        """
        if (len(sentences) < 3):
            return 0.0
        
        # Simple keyword consistency analysis : Extract meaningful words (nouns, adjectives)
        all_words = list()

        for sentence in sentences:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower())
            all_words.extend(words)
        
        if (len(all_words) < 10):
            return 0.0
        
        # Calculate how consistently keywords are used across sentences
        word_freq    = Counter(all_words)
        top_keywords = [word for word, count in word_freq.most_common(10) if count > 1]
        
        if not top_keywords:
            return 0.0
        
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
        chunks = list()
        words  = text.split()
        
        # Create overlapping chunks
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > 50):  
                chunk_sentences = self._split_sentences(chunk)
                
                if (len(chunk_sentences) >= 2):
                    embeddings = self._get_sentence_embeddings(chunk_sentences)
                    
                    if ((embeddings is not None) and (len(embeddings) >= 2)):
                        similarity_matrix = cosine_similarity(embeddings)
                        coherence         = self._calculate_coherence(similarity_matrix)
                        chunks.append(coherence)
        
        return chunks if chunks else [0.0]
    

    def _analyze_semantic_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze semantic patterns to determine RAW semantic score (0-1 scale)
        """
        # Check feature validity first
        required_features = ['coherence_score', 'consistency_score', 'repetition_score', 'topic_drift_score', 'coherence_variance']
    
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > 0]
    
        if (len(valid_features) < 3):
            # Low confidence if insufficient features
            return 0.5, 0.3  
    
        
        # Initialize ai_indicator list
        ai_indicators = list()

        # AI text often has very high coherence (too perfect)
        if (features['coherence_score'] > 0.7):
            # Suspiciously high coherence
            ai_indicators.append(0.8)  

        elif (features['coherence_score'] > 0.5):
            # Moderate coherence
            ai_indicators.append(0.5)

        else:
            # Low coherence - more human-like
            ai_indicators.append(0.2)  
        
        # Very high consistency suggests AI (unnaturally consistent)
        if (features['consistency_score'] > 0.8):
            ai_indicators.append(0.9)

        elif (features['consistency_score'] > 0.6):
            ai_indicators.append(0.6)

        else:
            ai_indicators.append(0.3)
        
        # High repetition suggests AI
        if (features['repetition_score'] > 0.3):
            ai_indicators.append(0.7)

        elif (features['repetition_score'] > 0.1):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # Very low topic drift suggests AI (stays too focused)
        if (features['topic_drift_score'] < 0.2):
            ai_indicators.append(0.8)

        elif (features['topic_drift_score'] < 0.4):
            ai_indicators.append(0.5)

        else:
            ai_indicators.append(0.3)
        
        # Low coherence variance across chunks suggests AI
        if (features['coherence_variance'] < 0.05):
            ai_indicators.append(0.7)

        elif (features['coherence_variance'] < 0.1):
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
        
        # Moderate coherence values might indicate mixing
        if (0.4 <= features['coherence_score'] <= 0.6):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        # High coherence variance suggests mixed content
        if (features['coherence_variance'] > 0.15):
            mixed_indicators.append(0.4)

        elif (features['coherence_variance'] > 0.1):
            mixed_indicators.append(0.2)

        else:
            mixed_indicators.append(0.0)
        
        # Inconsistent repetition patterns
        if (0.15 <= features['repetition_score'] <= 0.35):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        return min(0.3, np.mean(mixed_indicators)) if mixed_indicators else 0.0
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        return {"coherence_score"        : 0.5,
                "consistency_score"      : 0.5,
                "repetition_score"       : 0.0,
                "topic_drift_score"      : 0.5,
                "contextual_consistency" : 0.5,
                "avg_chunk_coherence"    : 0.5,
                "coherence_variance"     : 0.1,
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
