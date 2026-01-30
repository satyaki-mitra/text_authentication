# DEPENDENCIES
import re
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from collections import Counter
from config.enums import Domain
from config.schemas import MetricResult
from metrics.base_metric import BaseMetric
from models.model_manager import get_model_manager
from config.constants import linguistic_metric_params
from config.threshold_config import get_threshold_for_domain


class LinguisticMetric(BaseMetric):
    """
    Linguistic analysis using POS tagging, syntactic complexity, and grammatical patterns

    Mathematical Foundation:
    ------------------------
    - POS Entropy: H = -Σ p(pos) * log2(p(pos))
    - Dependency Tree Depth: Recursive maximum depth calculation
    - Syntactic Complexity: Weighted combination of average and max depths
    
    Measures:
    ---------
    - POS tag diversity and patterns
    - Syntactic complexity and sentence structure
    - Grammatical patterns and usage
    - Writing style analysis
    """
    def __init__(self):
        super().__init__(name        = "linguistic",
                         description = "POS tag diversity, syntactic complexity, and grammatical pattern analysis",
                        )

        self.nlp    = None
        self.params = linguistic_metric_params
    

    def initialize(self) -> bool:
        """
        Initialize the linguistic metric
        """
        try:
            logger.info("Initializing linguistic metric...")
            
            # Load spaCy model for linguistic analysis
            model_manager       = get_model_manager()
            self.nlp            = model_manager.load_model("linguistic_spacy")
            
            self.is_initialized = True
            logger.success("Linguistic metric initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize linguistic metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute linguistic analysis with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if ((not text) or (len(text.strip()) < self.params.MIN_TEXT_LENGTH_FOR_ANALYSIS)):
                return self._default_result(error = "Text too short for linguistic analysis")

            # Get domain-specific thresholds
            domain                                      = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                           = get_threshold_for_domain(domain)
            linguistic_thresholds                       = domain_thresholds.linguistic
            
            # Calculate comprehensive linguistic features
            features                                    = self._calculate_linguistic_features(text = text)
            
            # Calculate raw linguistic score (0-1 scale) with IMPROVED confidence
            raw_linguistic_score, confidence            = self._analyze_linguistic_patterns(features = features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            synthetic_prob, authentic_prob, hybrid_prob = self._apply_domain_thresholds(raw_score  = raw_linguistic_score, 
                                                                                        thresholds = linguistic_thresholds, 
                                                                                        features   = features,
                                                                                       )
            
            # Apply confidence multiplier from domain thresholds
            confidence                                 *= linguistic_thresholds.confidence_multiplier
            confidence                                  = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
            
            return MetricResult(metric_name           = self.name,
                                synthetic_probability = synthetic_prob,
                                authentic_probability = authentic_prob,
                                hybrid_probability    = hybrid_prob,
                                confidence            = confidence,
                                details               = {**features, 
                                                         'domain_used'        : domain.value,
                                                         'synthetic_threshold': linguistic_thresholds.synthetic_threshold,
                                                         'authentic_threshold': linguistic_thresholds.authentic_threshold,
                                                         'raw_score'          : raw_linguistic_score,
                                                        },
                               )
            
        except Exception as e:
            logger.error(f"Error in linguistic computation: {repr(e)}")
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
            range_width             = synthetic_threshold - authentic_threshold
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
        
        # Calculate hybrid probability based on linguistic variance
        hybrid_prob    = self._calculate_hybrid_probability(features)
        
        # Normalize to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob
        if (total > self.params.ZERO_TOLERANCE):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return synthetic_prob, authentic_prob, hybrid_prob
    

    def _calculate_linguistic_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive linguistic analysis features
        """
        if not self.nlp:
            return self._get_default_features()
        
        try:
            # Process text with spaCy
            doc                     = self.nlp(text)
            
            # Extract POS tags and dependencies
            pos_tags                = [token.pos_ for token in doc]
            
            # Calculate POS diversity and patterns
            pos_diversity           = self._calculate_pos_diversity(pos_tags = pos_tags)
            pos_entropy             = self._calculate_pos_entropy(pos_tags = pos_tags)
            
            # Calculate syntactic complexity
            syntactic_complexity    = self._calculate_syntactic_complexity(doc = doc)
            avg_sentence_complexity = self._calculate_sentence_complexity(doc = doc)
            
            # Analyze grammatical patterns
            grammatical_patterns    = self._analyze_grammatical_patterns(doc = doc)
            writing_style_score     = self._analyze_writing_style(doc = doc)
            
            # Chunk-based analysis for whole-text understanding
            chunk_complexities      = self._calculate_chunk_linguistics(text = text)
            
            avg_chunk_complexity    = np.mean(chunk_complexities) if chunk_complexities else 0.0
            complexity_variance     = np.var(chunk_complexities) if chunk_complexities else 0.0
            num_chunks              = len(chunk_complexities)
            
            # Calculate specific synthetic linguistic patterns
            synthetic_pattern_score = self._detect_synthetic_linguistic_patterns(doc = doc)
            
            # Count sentences for confidence calculation
            num_sentences           = len(list(doc.sents))
            
            return {"pos_diversity"           : round(pos_diversity, 4),
                    "pos_entropy"             : round(pos_entropy, 4),
                    "syntactic_complexity"    : round(syntactic_complexity, 4),
                    "avg_sentence_complexity" : round(avg_sentence_complexity, 4),
                    "grammatical_consistency" : round(grammatical_patterns['consistency'], 4),
                    "transition_word_usage"   : round(grammatical_patterns['transition_usage'], 4),
                    "passive_voice_ratio"     : round(grammatical_patterns['passive_ratio'], 4),
                    "writing_style_score"     : round(writing_style_score, 4),
                    "synthetic_pattern_score" : round(synthetic_pattern_score, 4),
                    "avg_chunk_complexity"    : round(avg_chunk_complexity, 4),
                    "complexity_variance"     : round(complexity_variance, 4),
                    "num_sentences"           : num_sentences,
                    "num_chunks_analyzed"     : num_chunks,
                   }
            
        except Exception as e:
            logger.warning(f"Linguistic analysis failed: {repr(e)}")
            return self._get_default_features()
    

    def _calculate_pos_diversity(self, pos_tags: List[str]) -> float:
        """
        Calculate POS tag diversity (type-token ratio for POS tags)
        
        Higher diversity = more varied sentence structures
        """
        if not pos_tags:
            return 0.0
        
        unique_pos = len(set(pos_tags))
        total_pos  = len(pos_tags)
        
        diversity  = unique_pos / total_pos
        return diversity
    

    def _calculate_pos_entropy(self, pos_tags: List[str]) -> float:
        """
        Calculate Shannon entropy of POS tag distribution
        
        Formula: H = -Σ p(pos) * log2(p(pos))
        
        Typical English: 2.5-3.5 bits
        """
        if (not pos_tags) or (len(pos_tags) < self.params.MIN_TAGS_FOR_ENTROPY):
            return 0.0
        
        pos_counts = Counter(pos_tags)
        total_tags = len(pos_tags)
        
        entropy    = 0.0

        for count in pos_counts.values():
            probability = count / total_tags

            if (probability > self.params.ZERO_TOLERANCE):
                entropy -= probability * np.log2(probability)
        
        return entropy
    

    def _calculate_syntactic_complexity(self, doc) -> float:
        """
        Calculate overall syntactic complexity based on dependency tree depth and structure
        
        Formula: complexity = (avg_depth * weight_avg) + (max_depth * weight_max)
        """
        complexities = list()
        
        for sent in doc.sents:
            # Calculate dependency tree depth
            depths = list()

            for token in sent:
                depth = self._calculate_dependency_depth(token)
                depths.append(depth)
            
            if depths:
                avg_depth  = np.mean(depths)
                max_depth  = np.max(depths)
                complexity = (avg_depth * self.params.COMPLEXITY_WEIGHT_AVG + max_depth * self.params.COMPLEXITY_WEIGHT_MAX)
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    

    def _calculate_dependency_depth(self, token, depth: int = 0) -> int:
        """
        Calculate dependency tree depth for a token (recursive)
        
        This is mathematically correct - traverses the parse tree to find maximum depth.
        """
        if not list(token.children):
            return depth
        
        child_depths = [self._calculate_dependency_depth(child, depth + 1) for child in token.children]

        return max(child_depths) if child_depths else depth
    

    def _calculate_sentence_complexity(self, doc) -> float:
        """
        Calculate average sentence complexity
        """
        complexities = list()
        
        for sent in doc.sents:
            # Simple complexity measure based on sentence length and structure
            words       = [token for token in sent if not token.is_punct]
            num_clauses = len([token for token in sent if token.dep_ in self.params.CLAUSE_MARKERS])
            
            if (len(words) > 0):
                complexity = (len(words) / self.params.WORDS_PER_COMPLEXITY_UNIT) + (num_clauses * self.params.CLAUSE_COMPLEXITY_FACTOR)
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    

    def _analyze_grammatical_patterns(self, doc) -> Dict[str, float]:
        """
        Analyze grammatical patterns and consistency
        """
        # Count different grammatical constructions
        passive_voice        = 0
        active_voice         = 0
        transition_words     = 0
        total_sentences      = 0
        
        for sent in doc.sents:
            total_sentences += 1
            sent_text        = sent.text.lower()
            
            # Check for passive voice patterns
            if (any(token.dep_ == self.params.PASSIVE_DEPENDENCY for token in sent)):
                passive_voice += 1
            
            else:
                active_voice += 1       
            
            # Count transition words
            for word in self.params.TRANSITION_WORDS_SET:
                if word in sent_text:
                    transition_words += 1
                    break
        
        # Calculate ratios
        passive_ratio    = passive_voice / total_sentences if total_sentences > 0 else 0.0
        transition_usage = transition_words / total_sentences if total_sentences > 0 else 0.0
        
        # Calculate consistency (lower variance in patterns)
        consistency      = 1.0 - min(1.0, abs(passive_ratio - self.params.IDEAL_PASSIVE_RATIO) + abs(transition_usage - self.params.IDEAL_TRANSITION_RATIO))
        
        return {'consistency'      : max(0.0, consistency),
                'passive_ratio'    : passive_ratio,
                'transition_usage' : transition_usage,
               }
    

    def _analyze_writing_style(self, doc) -> float:
        """
        Analyze writing style characteristics
        """
        style_indicators = list()
        
        # Sentence length variation
        sent_lengths     = [len([token for token in sent if not token.is_punct]) for sent in doc.sents]
        
        if sent_lengths:
            length_variation = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0.0
            # Moderate variation is more authentic-like
            style_score      = 1.0 - min(1.0, abs(length_variation - self.params.IDEAL_LENGTH_VARIATION))
            style_indicators.append(style_score)
        
        # Punctuation usage
        punct_ratio = len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0.0
        # Balanced punctuation is more authentic-like
        punct_score = 1.0 - min(1.0, abs(punct_ratio - self.params.IDEAL_PUNCTUATION_RATIO))
        style_indicators.append(punct_score)
        
        return np.mean(style_indicators) if style_indicators else 0.5
    

    def _detect_synthetic_linguistic_patterns(self, doc) -> float:
        """
        Detect synthetic-specific linguistic patterns
        """ 
        patterns_detected     = 0
        total_patterns        = 5
        
        # Pattern 1: Overuse of certain transition words
        transition_overuse    = self._check_transition_overuse(doc)
        
        if transition_overuse:
            patterns_detected += 1
        
        # Pattern 2: Unnatural POS sequences
        pos_sequences         = self._check_unnatural_pos_sequences(doc)
        
        if pos_sequences:
            patterns_detected += 1
        
        # Pattern 3: Overly consistent sentence structures
        structure_consistency = self._check_structure_consistency(doc)
        
        if structure_consistency:
            patterns_detected += 1
        
        # Pattern 4: Unusual grammatical constructions
        unusual_grammar       = self._check_unusual_grammar(doc)
        
        if unusual_grammar:
            patterns_detected += 1
        
        # Pattern 5: Repetitive phrasing
        repetitive_phrasing   = self._check_repetitive_phrasing(doc)
       
        if repetitive_phrasing:
            patterns_detected += 1
        
        return patterns_detected / total_patterns
    

    def _check_transition_overuse(self, doc) -> bool:
        """
        Check for overuse of transition words (common synthetic pattern)
        """
        transition_count = sum(1 for token in doc if token.lemma_.lower() in self.params.TRANSITION_WORDS_SET)
        
        # More than threshold of words being transitions is suspicious
        return transition_count / len(doc) > self.params.TRANSITION_OVERUSE_THRESHOLD if len(doc) > 0 else False
    

    def _check_unnatural_pos_sequences(self, doc) -> bool:
        """
        Check for unnatural POS tag sequences
        """
        pos_sequences = list()

        for sent in doc.sents:
            sent_pos = [token.pos_ for token in sent]
            pos_sequences.extend([(sent_pos[i], sent_pos[i+1]) for i in range(len(sent_pos)-1)])
        
        # Look for repetitive or unnatural sequences
        if not pos_sequences:
            return False
        
        sequence_counts  = Counter(pos_sequences)
        most_common_freq = max(sequence_counts.values()) / len(pos_sequences) if pos_sequences else 0
        
        # High frequency of specific sequences suggests synthetic
        return (most_common_freq > self.params.POS_SEQUENCE_FREQ_THRESHOLD)
    

    def _check_structure_consistency(self, doc) -> bool:
        """
        Check for overly consistent sentence structures
        """
        sent_structures = list()
        
        for sent in doc.sents:
            # Simple structure representation
            structure = tuple(token.dep_ for token in sent if token.dep_ not in ['punct', 'det'])
            sent_structures.append(structure)
        
        if (len(sent_structures) < self.params.MIN_SENTENCES_FOR_STRUCTURE):
            return False
        
        # Calculate structure similarity
        unique_structures = len(set(sent_structures))
        similarity_ratio  = unique_structures / len(sent_structures)
        
        # Low diversity suggests synthetic
        return (similarity_ratio < self.params.STRUCTURE_DIVERSITY_THRESHOLD)
    

    def _check_unusual_grammar(self, doc) -> bool:
        """
        Check for unusual grammatical constructions
        """
        unusual_constructions = 0
        
        for token in doc:
            # Check for unusual dependency relations i.e. less common relations
            if token.dep_ in self.params.UNUSUAL_DEPENDENCIES:  
                unusual_constructions += 1
        
        # More than threshold unusual constructions is suspicious
        return (unusual_constructions / len(doc) > self.params.UNUSUAL_CONSTRUCTION_THRESHOLD) if (len(doc) > 0) else False
    

    def _check_repetitive_phrasing(self, doc) -> bool:
        """
        Check for repetitive phrasing patterns
        """
        phrases = list()

        for sent in doc.sents:
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in sent.noun_chunks]
            phrases.extend(noun_phrases)
        
        if not phrases:
            return False
        
        phrase_counts    = Counter(phrases)
        repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
        
        # High repetition suggests synthetic
        return (repeated_phrases / len(phrases) > self.params.REPETITIVE_PHRASING_THRESHOLD)
    

    def _calculate_chunk_linguistics(self, text: str) -> List[float]:
        """
        Calculate linguistic features across text chunks
        """
        complexities = list()
        words        = text.split()
        chunk_size   = self.params.CHUNK_SIZE_WORDS
        overlap      = int(chunk_size * self.params.CHUNK_OVERLAP_RATIO)
        step         = max(1, chunk_size - overlap)

        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if (len(chunk) > self.params.MIN_CHUNK_LENGTH):
                try:
                    chunk_doc = self.nlp(chunk)
                    
                    # Check if processing was successful (CORRECTED - now uses constant)
                    if (chunk_doc and (len(list(chunk_doc.sents)) > self.params.MIN_SENTENCES_FOR_CHUNK_VALIDITY)):
                        complexity = self._calculate_syntactic_complexity(chunk_doc)
                        complexities.append(complexity)
                
                except Exception as e:
                    logger.debug(f"Chunk linguistic analysis failed: {e}")
                    continue
        
        return complexities
    

    def _analyze_linguistic_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze linguistic patterns to determine RAW linguistic score (0-1 scale) 
        
        Returns:
        --------
        (raw_score, confidence) where:
        - raw_score: Higher = more synthetic-like
        - confidence: Based on sample size and agreement
        """
        # Check feature validity first
        required_features = ['pos_diversity', 'pos_entropy', 'syntactic_complexity', 'grammatical_consistency', 'transition_word_usage', 'synthetic_pattern_score', 'complexity_variance']
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > self.params.ZERO_TOLERANCE]
        
        if (len(valid_features) < self.params.MIN_REQUIRED_FEATURES):
            # Low confidence if insufficient features
            return self.params.NEUTRAL_PROBABILITY, self.params.LOW_FEATURE_CONFIDENCE

        # Initialize synthetic_indicator list 
        synthetic_indicators = list()
        
        # Low POS diversity suggests synthetic
        if (features['pos_diversity'] < self.params.POS_DIVERSITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)

        elif (features['pos_diversity'] < self.params.POS_DIVERSITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)

        
        # Low POS entropy suggests templated / synthetic language
        if (features['pos_entropy'] < self.params.POS_ENTROPY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)
        
        elif (features['pos_entropy'] < self.params.POS_ENTROPY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)
        
        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Low syntactic complexity suggests synthetic
        if (features['syntactic_complexity'] < self.params.SYNTACTIC_COMPLEXITY_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        elif (features['syntactic_complexity'] < self.params.SYNTACTIC_COMPLEXITY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.VERY_LOW_SYNTHETIC_WEIGHT)
        
        # High grammatical consistency suggests synthetic (unnaturally consistent)
        if (features['grammatical_consistency'] > self.params.GRAMMATICAL_CONSISTENCY_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.STRONG_SYNTHETIC_WEIGHT)

        elif (features['grammatical_consistency'] > self.params.GRAMMATICAL_CONSISTENCY_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.LOW_SYNTHETIC_WEIGHT)
        
        # High transition word usage suggests synthetic
        if (features['transition_word_usage'] > self.params.TRANSITION_USAGE_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)
        
        elif (features['transition_word_usage'] > self.params.TRANSITION_USAGE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.VERY_LOW_SYNTHETIC_WEIGHT)
        
        # High synthetic pattern score suggests synthetic
        if (features['synthetic_pattern_score'] > self.params.SYNTHETIC_PATTERN_HIGH_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        elif (features['synthetic_pattern_score'] > self.params.SYNTHETIC_PATTERN_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.MODERATE_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.MINIMAL_SYNTHETIC_WEIGHT)
        
        # Low complexity variance suggests synthetic
        if (features['complexity_variance'] < self.params.COMPLEXITY_VARIANCE_LOW_THRESHOLD):
            synthetic_indicators.append(self.params.MEDIUM_SYNTHETIC_WEIGHT)

        elif (features['complexity_variance'] < self.params.COMPLEXITY_VARIANCE_MEDIUM_THRESHOLD):
            synthetic_indicators.append(self.params.WEAK_SYNTHETIC_WEIGHT)

        else:
            synthetic_indicators.append(self.params.VERY_LOW_SYNTHETIC_WEIGHT)
        
        # Calculate raw score
        raw_score  = np.mean(synthetic_indicators) if synthetic_indicators else self.params.NEUTRAL_PROBABILITY
        
        # Factor 1: Agreement between indicators (lower std = higher confidence)
        agreement_confidence = 1.0 - min(1.0, np.std(synthetic_indicators) / self.params.CONFIDENCE_STD_NORMALIZER)
        
        # Factor 2: Sample size adequacy
        num_sentences        = features.get('num_sentences', 0)
        num_chunks           = features.get('num_chunks_analyzed', 0)
        sentence_confidence  = min(1.0, num_sentences / self.params.MIN_SENTENCES_FOR_CONFIDENCE)
        chunk_confidence     = min(1.0, num_chunks / self.params.MIN_CHUNKS_FOR_CONFIDENCE)
        sample_confidence    = (sentence_confidence + chunk_confidence) / 2.0
        
        # Combine factors
        confidence           = (self.params.CONFIDENCE_BASE + self.params.CONFIDENCE_STD_FACTOR * agreement_confidence + self.params.CONFIDENCE_SAMPLE_FACTOR * sample_confidence)
        
        confidence           = max(self.params.MIN_CONFIDENCE, min(self.params.MAX_CONFIDENCE, confidence))
        
        return raw_score, confidence
    

    def _calculate_hybrid_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of hybrid synthetic/authentic content
        """
        hybrid_indicators = list()
        
        # Moderate POS diversity might indicate mixing
        if (self.params.POS_DIVERSITY_MIXED_MIN <= features['pos_diversity'] <= self.params.POS_DIVERSITY_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # High complexity variance suggests mixed content
        if (features['complexity_variance'] > self.params.COMPLEXITY_VARIANCE_HIGH_THRESHOLD):
            hybrid_indicators.append(self.params.MODERATE_HYBRID_WEIGHT)

        elif (features['complexity_variance'] > self.params.COMPLEXITY_VARIANCE_MEDIUM_THRESHOLD):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        # Inconsistent synthetic pattern detection
        if (self.params.SYNTHETIC_PATTERN_MIXED_MIN <= features['synthetic_pattern_score'] <= self.params.SYNTHETIC_PATTERN_MIXED_MAX):
            hybrid_indicators.append(self.params.WEAK_HYBRID_WEIGHT)

        else:
            hybrid_indicators.append(self.params.MINIMAL_HYBRID_WEIGHT)
        
        hybrid_prob = np.mean(hybrid_indicators) if hybrid_indicators else 0.0
        return min(self.params.MAX_HYBRID_PROBABILITY, hybrid_prob)
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        return {"pos_diversity"           : self.params.DEFAULT_POS_DIVERSITY,
                "pos_entropy"             : self.params.DEFAULT_POS_ENTROPY,
                "syntactic_complexity"    : self.params.DEFAULT_SYNTACTIC_COMPLEXITY,
                "avg_sentence_complexity" : self.params.DEFAULT_SENTENCE_COMPLEXITY,
                "grammatical_consistency" : self.params.DEFAULT_GRAMMATICAL_CONSISTENCY,
                "transition_word_usage"   : self.params.DEFAULT_TRANSITION_USAGE,
                "passive_voice_ratio"     : self.params.DEFAULT_PASSIVE_RATIO,
                "writing_style_score"     : self.params.DEFAULT_WRITING_STYLE_SCORE,
                "synthetic_pattern_score" : self.params.DEFAULT_SYNTHETIC_PATTERN_SCORE,
                "avg_chunk_complexity"    : self.params.DEFAULT_CHUNK_COMPLEXITY,
                "complexity_variance"     : self.params.DEFAULT_COMPLEXITY_VARIANCE,
                "num_sentences"           : 0,
                "num_chunks_analyzed"     : 0,
               }
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.nlp = None
        super().cleanup()



# Export
__all__ = ["LinguisticMetric"]