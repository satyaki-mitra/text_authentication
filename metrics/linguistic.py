# DEPENDENCIES
import re
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from collections import Counter
from config.threshold_config import Domain
from metrics.base_metric import BaseMetric
from metrics.base_metric import MetricResult
from models.model_manager import get_model_manager
from config.threshold_config import get_threshold_for_domain


class LinguisticMetric(BaseMetric):
    """
    Linguistic analysis using POS tagging, syntactic complexity, and grammatical patterns
    
    Measures (Aligned with Documentation):
    - POS tag diversity and patterns
    - Syntactic complexity and sentence structure
    - Grammatical patterns and usage
    - Writing style analysis
    """
    def __init__(self):
        super().__init__(name        = "linguistic",
                         description = "POS tag diversity, syntactic complexity, and grammatical pattern analysis",
                        )
        self.nlp = None
    

    def initialize(self) -> bool:
        """
        Initialize the linguistic metric
        """
        try:
            logger.info("Initializing linguistic metric...")
            
            # Load spaCy model for linguistic analysis
            model_manager = get_model_manager()
            self.nlp      = model_manager.load_model("linguistic_spacy")
            
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
            if ((not text) or (len(text.strip()) < 50)):
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for linguistic analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                           = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                = get_threshold_for_domain(domain)
            linguistic_thresholds            = domain_thresholds.linguistic
            
            # Calculate comprehensive linguistic features
            features                         = self._calculate_linguistic_features(text)
            
            # Calculate raw linguistic score (0-1 scale)
            raw_linguistic_score, confidence = self._analyze_linguistic_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob  = self._apply_domain_thresholds(raw_linguistic_score, linguistic_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                      *= linguistic_thresholds.confidence_multiplier
            confidence                       = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : linguistic_thresholds.ai_threshold,
                                                     'human_threshold' : linguistic_thresholds.human_threshold,
                                                     'raw_score'       : raw_linguistic_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in linguistic computation: {repr(e)}")
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
        ai_threshold    = thresholds.ai_threshold
        human_threshold = thresholds.human_threshold

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
        
        # Calculate mixed probability based on linguistic variance
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob
        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob
    

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
            dependencies            = [token.dep_ for token in doc]
            
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
            chunk_features          = self._calculate_chunk_linguistics(text       = text, 
                                                                        chunk_size = 200,
                                                                       )
            
            # Calculate specific AI linguistic patterns
            ai_pattern_score        = self._detect_ai_linguistic_patterns(doc = doc)
            
            return {"pos_diversity"           : round(pos_diversity, 4),
                    "pos_entropy"             : round(pos_entropy, 4),
                    "syntactic_complexity"    : round(syntactic_complexity, 4),
                    "avg_sentence_complexity" : round(avg_sentence_complexity, 4),
                    "grammatical_consistency" : round(grammatical_patterns['consistency'], 4),
                    "transition_word_usage"   : round(grammatical_patterns['transition_usage'], 4),
                    "passive_voice_ratio"     : round(grammatical_patterns['passive_ratio'], 4),
                    "writing_style_score"     : round(writing_style_score, 4),
                    "ai_pattern_score"        : round(ai_pattern_score, 4),
                    "avg_chunk_complexity"    : round(np.mean(chunk_features['complexities']) if chunk_features['complexities'] else 0.0, 4),
                    "complexity_variance"     : round(np.var(chunk_features['complexities']) if chunk_features['complexities'] else 0.0, 4),
                    "num_sentences"           : len(list(doc.sents)),
                    "num_chunks_analyzed"     : len(chunk_features['complexities']),
                   }
            
        except Exception as e:
            logger.warning(f"Linguistic analysis failed: {repr(e)}")
            return self._get_default_features()
    

    def _calculate_pos_diversity(self, pos_tags: List[str]) -> float:
        """
        Calculate POS tag diversity : Higher diversity = more varied sentence structures
        """
        if not pos_tags:
            return 0.0
        
        unique_pos = len(set(pos_tags))
        total_pos  = len(pos_tags)
        
        diversity  = unique_pos / total_pos
        return diversity
    

    def _calculate_pos_entropy(self, pos_tags: List[str]) -> float:
        """
        Calculate entropy of POS tag distribution
        """
        if not pos_tags:
            return 0.0
        
        pos_counts = Counter(pos_tags)
        total_tags = len(pos_tags)
        
        entropy = 0.0
        for count in pos_counts.values():
            probability = count / total_tags
            entropy    -= probability * np.log2(probability)
        
        return entropy
    

    def _calculate_syntactic_complexity(self, doc) -> float:
        """
        Calculate overall syntactic complexity : based on dependency tree depth and structure
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
                complexity = (avg_depth + max_depth) / 2.0
                complexities.append(complexity)
        
        return np.mean(complexities) if complexities else 0.0
    

    def _calculate_dependency_depth(self, token, depth: int = 0) -> int:
        """
        Calculate dependency tree depth for a token
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
            num_clauses = len([token for token in sent if token.dep_ in ['cc', 'mark']])
            
            if (len(words) > 0):
                complexity = (len(words) / 10.0) + (num_clauses * 0.5)

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
        
        transition_words_set = {'however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally', 'nevertheless', 'nonetheless', 'thus', 'hence'}
        
        for sent in doc.sents:
            total_sentences += 1
            sent_text        = sent.text.lower()
            
            # Check for passive voice patterns
            if (any(token.dep_ == 'nsubjpass' for token in sent)):
                passive_voice += 1
            
            else:
                active_voice += 1
            
            # Count transition words
            for word in transition_words_set:
                if word in sent_text:
                    transition_words += 1
                    break
        
        # Calculate ratios
        passive_ratio    = passive_voice / total_sentences if total_sentences > 0 else 0.0
        transition_usage = transition_words / total_sentences if total_sentences > 0 else 0.0
        
        # Calculate consistency (lower variance in patterns)
        consistency      = 1.0 - min(1.0, abs(passive_ratio - 0.3) + abs(transition_usage - 0.2))
        
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
        sent_lengths = [len([token for token in sent if not token.is_punct]) for sent in doc.sents]
        
        if sent_lengths:
            length_variation = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0.0
            # Moderate variation is more human-like
            style_score      = 1.0 - min(1.0, abs(length_variation - 0.5))

            style_indicators.append(style_score)
        
        # Punctuation usage
        punct_ratio = len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0.0
        # Balanced punctuation is more human-like
        punct_score = 1.0 - min(1.0, abs(punct_ratio - 0.1))

        style_indicators.append(punct_score)
        
        return np.mean(style_indicators) if style_indicators else 0.5
    

    def _detect_ai_linguistic_patterns(self, doc) -> float:
        """
        Detect AI-specific linguistic patterns
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
        Check for overuse of transition words (common AI pattern)
        """
        transition_words = {'however', 'therefore', 'moreover', 'furthermore', 'additionally'}
        transition_count = sum(1 for token in doc if token.lemma_.lower() in transition_words)
        
        # More than 5% of words being transitions is suspicious
        return transition_count / len(doc) > 0.05 if len(doc) > 0 else False
    

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
        
        # High frequency of specific sequences suggests AI
        return (most_common_freq > 0.1)
    

    def _check_structure_consistency(self, doc) -> bool:
        """
        Check for overly consistent sentence structures
        """
        sent_structures = list()
        
        for sent in doc.sents:
            # Simple structure representation
            structure = tuple(token.dep_ for token in sent if token.dep_ not in ['punct', 'det'])
            sent_structures.append(structure)
        
        if (len(sent_structures) < 3):
            return False
        
        # Calculate structure similarity
        unique_structures = len(set(sent_structures))
        similarity_ratio  = unique_structures / len(sent_structures)
        
        # Low diversity suggests AI
        return (similarity_ratio < 0.5)
    

    def _check_unusual_grammar(self, doc) -> bool:
        """
        Check for unusual grammatical constructions
        """
        unusual_constructions = 0
        
        for token in doc:
            # Check for unusual dependency relations i.e. less common relations
            if token.dep_ in ['attr', 'oprd']:  
                unusual_constructions += 1
        
        # More than 2% unusual constructions is suspicious
        return (unusual_constructions / len(doc) > 0.02) if (len(doc) > 0) else False
    

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
        
        # High repetition suggests AI
        return (repeated_phrases / len(phrases) > 0.3)
    

    def _calculate_chunk_linguistics(self, text: str, chunk_size: int = 200) -> Dict[str, List[float]]:
        """
        Calculate linguistic features across text chunks
        """
        complexities = list()
        words        = text.split()
        
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            
            if (len(chunk) > 50):
                try:
                    chunk_doc = self.nlp(chunk)
                    
                    # Check if processing was successful
                    if (chunk_doc and (len(list(chunk_doc.sents)) > 0)):
                        complexity = self._calculate_syntactic_complexity(chunk_doc)
                        complexities.append(complexity)
                
                except Exception as e:
                    logger.debug(f"Chunk linguistic analysis failed: {e}")
                    continue
        
        return {'complexities': complexities}
    

    def _analyze_linguistic_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze linguistic patterns to determine RAW linguistic score (0-1 scale) : Higher score = more AI-like
        """
        # Check feature validity first
        required_features = ['pos_diversity', 'syntactic_complexity', 'grammatical_consistency', 'transition_word_usage', 'ai_pattern_score', 'complexity_variance']
        
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > 0]
        
        if (len(valid_features) < 4):
            # Low confidence if insufficient features
            return 0.5, 0.3  

        # Initialize ai_indicator list 
        ai_indicators = list()
        
        # Low POS diversity suggests AI
        if (features['pos_diversity'] < 0.3):
            ai_indicators.append(0.8)

        elif (features['pos_diversity'] < 0.5):
            ai_indicators.append(0.6)

        else:
            ai_indicators.append(0.2)
        
        # Low syntactic complexity suggests AI
        if (features['syntactic_complexity'] < 2.0):
            ai_indicators.append(0.7)

        elif (features['syntactic_complexity'] < 3.0):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # High grammatical consistency suggests AI (unnaturally consistent)
        if (features['grammatical_consistency'] > 0.8):
            ai_indicators.append(0.9)

        elif (features['grammatical_consistency'] > 0.6):
            ai_indicators.append(0.5)

        else:
            ai_indicators.append(0.3)
        
        # High transition word usage suggests AI
        if (features['transition_word_usage'] > 0.3):
            ai_indicators.append(0.7)
        
        elif (features['transition_word_usage'] > 0.15):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # High AI pattern score suggests AI
        if (features['ai_pattern_score'] > 0.6):
            ai_indicators.append(0.8)

        elif (features['ai_pattern_score'] > 0.3):
            ai_indicators.append(0.5)

        else:
            ai_indicators.append(0.2)
        
        # Low complexity variance suggests AI
        if (features['complexity_variance'] < 0.1):
            ai_indicators.append(0.7)

        elif (features['complexity_variance'] < 0.3):
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
        
        # Moderate POS diversity might indicate mixing
        if (0.35 <= features['pos_diversity'] <= 0.55):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        # High complexity variance suggests mixed content
        if (features['complexity_variance'] > 0.5):
            mixed_indicators.append(0.4)

        elif (features['complexity_variance'] > 0.3):
            mixed_indicators.append(0.2)

        else:
            mixed_indicators.append(0.0)
        
        # Inconsistent AI pattern detection
        if (0.2 <= features['ai_pattern_score'] <= 0.6):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        return min(0.3, np.mean(mixed_indicators)) if mixed_indicators else 0.0
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        return {"pos_diversity"           : 0.5,
                "pos_entropy"             : 2.5,
                "syntactic_complexity"    : 2.5,
                "avg_sentence_complexity" : 2.0,
                "grammatical_consistency" : 0.5,
                "transition_word_usage"   : 0.1,
                "passive_voice_ratio"     : 0.2,
                "writing_style_score"     : 0.5,
                "ai_pattern_score"        : 0.3,
                "avg_chunk_complexity"    : 2.5,
                "complexity_variance"     : 0.2,
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