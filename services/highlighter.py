# DEPENDENCIES
import re
from typing import List
from typing import Dict
from typing import Tuple
from loguru import logger
from typing import Optional
from config.enums import Domain
from config.schemas import MetricResult
from config.schemas import EnsembleResult
from processors.text_processor import TextProcessor
from config.threshold_config import ConfidenceLevel
from config.schemas import HighlightedSentenceResult
from config.threshold_config import MetricThresholds
from config.threshold_config import get_confidence_level
from services.ensemble_classifier import EnsembleClassifier
from config.threshold_config import get_threshold_for_domain
from config.threshold_config import get_active_metric_weights


class TextHighlighter:
    """
    Generates sentence-level highlighting with ensemble results integration
    
    FEATURES:
    - Sentence-level highlighting with confidence scores
    - Domain-aware calibration
    - Ensemble-assisted probability aggregation
    - Hybrid content detection
    - Explainable tooltips
    """
    # Color thresholds - 4 categories
    COLOR_THRESHOLDS             = [(0.00, 0.40, "authentic", "#d1fae5", "Likely authentically written"),    # Authentic: Synthetic probability < 0.4
                                    (0.40, 0.60, "uncertain", "#fef3c7", "Uncertain authorship"),            # Uncertain: 0.4 ≤ Synthetic probability < 0.6
                                    (0.60, 0.80, "hybrid", "#e9d5ff", "Mixed synthetic/authentic content"),  # Hybrid: 0.6 ≤ Synthetic probability < 0.8 OR explicit hybrid detection
                                    (0.80, 1.01, "synthetic", "#fee2e2", "Likely synthetically generated"),  # Synthetic: Synthetic probability ≥ 0.8
                                   ]
    
    # Hybrid detection thresholds
    HYBRID_PROB_THRESHOLD        = 0.25  # Minimum hybrid probability to classify as hybrid

    def __init__(self, domain: Domain = Domain.GENERAL, ensemble_classifier: Optional[EnsembleClassifier] = None):
        """
        Initialize text highlighter with ENSEMBLE INTEGRATION
        
        Arguments:
        ----------
            domain                    { Domain }       : Text domain for adaptive thresholding

            ensemble_classifier { EnsembleClassifier } : Optional ensemble for sentence-level analysis
        """
        self.text_processor     = TextProcessor()
        self.domain             = domain
        self.domain_thresholds  = get_threshold_for_domain(domain)
        self.ensemble           = ensemble_classifier or self._create_default_ensemble()
    

    def _create_default_ensemble(self) -> EnsembleClassifier:
        """
        Create default ensemble classifier with proper error handling
        """
        try:
            return EnsembleClassifier(primary_method  = "confidence_calibrated",
                                      fallback_method = "domain_weighted",
                                     )
        except Exception as e:
            logger.warning(f"Failed to create default ensemble: {e}. Using fallback mode.")
            return EnsembleClassifier(primary_method  = "domain_weighted",
                                      fallback_method = "simple_average",
                                     )


    def generate_highlights(self, text: str, metric_results: Dict[str, MetricResult], ensemble_result: Optional[EnsembleResult] = None,
                            enabled_metrics: Optional[Dict[str, bool]] = None, use_sentence_level: bool = True) -> List[HighlightedSentenceResult]:
        """
        Generate sentence-level highlights with ensemble integration
        
        Arguments:
        ----------
            text                    { str }       : Original text

            metric_results          { dict }      : Results from all metrics
            
            ensemble_result    { EnsembleResult } : Optional document-level ensemble result
            
            enabled_metrics         { dict }      : Dict of metric_name -> is_enabled
            
            use_sentence_level      { bool }      : Whether to compute sentence-level probabilities
            
        Returns:
        --------
                         { list }                 : List of HighlightedSentenceResult objects
        """
        try:
            # Validate inputs
            if not text or not text.strip():
                return self._handle_empty_text(text            = text, 
                                               metric_results  = metric_results, 
                                               ensemble_result = ensemble_result,
                                              )
            
            # Get domain-appropriate weights for enabled metrics
            if enabled_metrics is None:
                enabled_metrics = {name: True for name in metric_results.keys()}
            
            weights   = get_active_metric_weights(self.domain, enabled_metrics)
            
            # Split text into sentences with error handling
            sentences = self._split_sentences_with_fallback(text = text)
            
            if not sentences:
                return self._handle_no_sentences(text, metric_results, ensemble_result)
            
            # Calculate probabilities for each sentence using ENSEMBLE METHODS
            highlighted_sentences = list()
            
            for idx, sentence in enumerate(sentences):
                try:
                    if use_sentence_level:
                        # Use ensemble for sentence-level analysis
                        synthetic_prob, authentic_prob, hybrid_prob, confidence, breakdown = self._calculate_sentence_ensemble_probability(sentence        = sentence,
                                                                                                                                           metric_results  = metric_results,
                                                                                                                                           weights         = weights,
                                                                                                                                           ensemble_result = ensemble_result,
                                                                                                                                          )
                    else:
                        # Use document-level ensemble probabilities
                        synthetic_prob, authentic_prob, hybrid_prob, confidence, breakdown = self._get_document_ensemble_probability(ensemble_result = ensemble_result,
                                                                                                                                     metric_results  = metric_results,
                                                                                                                                     weights         = weights,
                                                                                                                                    )
                    
                    # Apply domain-specific adjustments with limits
                    synthetic_prob                       = self._apply_domain_specific_adjustments(sentence        = sentence,
                                                                                                   synthetic_prob  = synthetic_prob,
                                                                                                   sentence_length = len(sentence.split()),
                                                                                                  )
                    
                    # Determine if this is hybrid content
                    is_hybrid_content                    = self._is_hybrid_content(synthetic_prob = synthetic_prob,
                                                                                   hybrid_prob    = hybrid_prob,
                                                                                   confidence     = confidence,
                                                                                  )
                    
                    # Get confidence level
                    confidence_level                     = get_confidence_level(confidence)
                    
                    # Get color class (consider hybrid content)
                    color_class, color_hex, tooltip_base = self._get_color_for_probability(synthetic_prob    = synthetic_prob,
                                                                                           is_hybrid_content = is_hybrid_content,
                                                                                           hybrid_prob       = hybrid_prob,
                                                                                          )
                    
                    # Generate enhanced tooltip
                    tooltip                              = self._generate_ensemble_tooltip(sentence          = sentence,
                                                                                           synthetic_prob    = synthetic_prob,
                                                                                           authentic_prob    = authentic_prob,
                                                                                           hybrid_prob       = hybrid_prob,
                                                                                           confidence        = confidence,
                                                                                           confidence_level  = confidence_level,
                                                                                           tooltip_base      = tooltip_base,
                                                                                           breakdown         = breakdown,
                                                                                           is_hybrid_content = is_hybrid_content,
                                                                                          )
                    
                    highlighted_sentences.append(HighlightedSentenceResult(text                  = sentence,
                                                                           synthetic_probability = synthetic_prob,
                                                                           authentic_probability = authentic_prob,
                                                                           hybrid_probability    = hybrid_prob,
                                                                           confidence            = confidence,
                                                                           confidence_level      = confidence_level,
                                                                           color_class           = color_class,
                                                                           tooltip               = tooltip,
                                                                           index                 = idx,
                                                                           is_hybrid_content     = is_hybrid_content,
                                                                           metric_breakdown      = breakdown,
                                                                          )
                                                )
                
                except Exception as e:
                    logger.warning(f"Failed to process sentence {idx}: {e}")
                    # Add fallback sentence
                    highlighted_sentences.append(self._create_fallback_sentence(sentence, idx))
            
            return highlighted_sentences
        
        except Exception as e:
            logger.error(f"Highlight generation failed: {e}")
            return self._create_error_fallback(text, metric_results)


    def _handle_empty_text(self, text: str, metric_results: Dict[str, MetricResult], ensemble_result: Optional[EnsembleResult]) -> List[HighlightedSentenceResult]:
        """
        Handle empty input text
        """
        if ensemble_result:
            return [self._create_fallback_sentence(text           = "No text content",
                                                   index          = 0,
                                                   synthetic_prob = ensemble_result.synthetic_probability,
                                                   authentic_prob = ensemble_result.authentic_probability,
                                                  )
                   ]

        return [self._create_fallback_sentence("No text content", 0)]


    def _handle_no_sentences(self, text: str, metric_results: Dict[str, MetricResult], ensemble_result: Optional[EnsembleResult]) -> List[HighlightedSentenceResult]:
        """
        Handle case where no sentences could be extracted
        """
        if text and text.strip():
            # Treat entire text as one sentence
            return [self._create_fallback_sentence(text.strip(), 0)]
        
        return [self._create_fallback_sentence("No processable content", 0)]


    def _create_fallback_sentence(self, text: str, index: int, synthetic_prob: float = 0.5, authentic_prob: float = 0.5) -> HighlightedSentenceResult:
        """
        Create a fallback sentence when processing fails
        """
        confidence_level             = get_confidence_level(0.3)
        color_class, _, tooltip_base = self._get_color_for_probability(synthetic_prob    = synthetic_prob,
                                                                       is_hybrid_content = False,
                                                                       hybrid_prob       = 0.0,
                                                                      )
        
        return HighlightedSentenceResult(text                  = text,
                                         synthetic_probability = synthetic_prob,
                                         authentic_probability = authentic_prob,
                                         hybrid_probability    = 0.0,
                                         confidence            = 0.3,
                                         confidence_level      = confidence_level,
                                         color_class           = color_class,
                                         tooltip               = f"Fallback: {tooltip_base}\nProcessing failed for this sentence",
                                         index                 = index,
                                         is_hybrid_content     = False,
                                         metric_breakdown      = {"fallback": synthetic_prob},
                                        )


    def _create_error_fallback(self, text: str, metric_results: Dict[str, MetricResult]) -> List[HighlightedSentenceResult]:
        """
        Create fallback when entire processing fails
        """
        return [HighlightedSentenceResult(text                  = text[:100] + "..." if len(text) > 100 else text,
                                          synthetic_probability = 0.5,
                                          authentic_probability = 0.5,
                                          hybrid_probability    = 0.0,
                                          confidence            = 0.1,
                                          confidence_level      = get_confidence_level(0.1),
                                          color_class           = "uncertain",
                                          tooltip               = "Error in text processing",
                                          index                 = 0,
                                          is_hybrid_content     = False,
                                          metric_breakdown      = {"error": 0.5},
                                         )
               ]
 

    def _split_sentences_with_fallback(self, text: str) -> List[str]:
        """
        Split text into sentences with comprehensive fallback handling
        """
        try:
            sentences          = self.text_processor.split_sentences(text)
            filtered_sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]
            
            if filtered_sentences:
                return filtered_sentences
            
            # Fallback: split by common sentence endings
            fallback_sentences = re.split(r'[.!?]+', text)
            fallback_sentences = [s.strip() for s in fallback_sentences if len(s.strip()) >= 3]
            
            if fallback_sentences:
                return fallback_sentences
            
            # Ultimate fallback: treat as single sentence if meaningful
            if text.strip():
                return [text.strip()]
            
            return []
            
        except Exception as e:
            logger.warning(f"Sentence splitting failed, using fallback: {e}")
            # Return text as single sentence
            return [text] if text.strip() else []


    def _calculate_sentence_ensemble_probability(self, sentence: str, metric_results: Dict[str, MetricResult], weights: Dict[str, float], ensemble_result: Optional[EnsembleResult] = None) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Calculate sentence probabilities using ensemble methods with domain calibration
        """
        sentence_length = len(sentence.split())

        # Handling very short sentences – do not force neutral, but reduce confidence
        if (sentence_length < 3):
            base_synthetic_prob = 0.5
            base_confidence     = 0.2
            breakdown           = {"short_sentence": base_synthetic_prob}

            for name, result in metric_results.items():
                if (result.error is None and weights.get(name, 0.0) > 0):
                    base_synthetic_prob = result.synthetic_probability
                    breakdown[name]     = base_synthetic_prob
                    break

            return (base_synthetic_prob,
                    1.0 - base_synthetic_prob,
                    0.0,
                    base_confidence,
                    breakdown
                   )

        # Build sentence-level metric results
        sentence_metric_results = dict()
        breakdown               = dict()

        for name, doc_result in metric_results.items():
            if doc_result.error is not None:
                continue

            try:
                sentence_prob                 = self._compute_sentence_metric(metric_name = name,
                                                                              sentence    = sentence,
                                                                              result      = doc_result,
                                                                              weight      = weights.get(name, 0.0),
                                                                             )

                sentence_metric_results[name] = self._create_sentence_metric_result(metric_name     = name,
                                                                                    synthetic_prob  = sentence_prob,
                                                                                    doc_result      = doc_result,
                                                                                    sentence_length = sentence_length,
                                                                                   )

                breakdown[name]               = sentence_prob

            except Exception as e:
                logger.warning(f"Metric {name} failed for sentence: {e}")
                breakdown[name] = doc_result.synthetic_probability

        # Ensemble aggregation (PRIMARY PATH)
        if sentence_metric_results:
            try:
                ensemble_sentence_result = self.ensemble.predict(metric_results = sentence_metric_results,
                                                                 domain         = self.domain,
                                                                )

                return (ensemble_sentence_result.synthetic_probability,
                        ensemble_sentence_result.authentic_probability,
                        ensemble_sentence_result.hybrid_probability,   
                        ensemble_sentence_result.overall_confidence,
                        breakdown,
                       )

            except Exception as e:
                logger.warning(f"Sentence ensemble failed: {e}")

        # Fallback: weighted average aggregation
        return self._fallback_weighted_probability(metric_results, weights, breakdown)


    def _compute_sentence_metric(self, metric_name: str, sentence: str, result: MetricResult, weight: float) -> float:
        """
        Compute metric probability for a single sentence using domain-specific thresholds
        """
        sentence_length   = len(sentence.split())
        
        # Get domain-specific threshold for this metric
        metric_thresholds = getattr(self.domain_thresholds, metric_name, None)
        
        if not metric_thresholds:
            return result.synthetic_probability
        
        # Base probability from document-level result
        base_prob         = result.synthetic_probability
        
        # Apply domain-aware sentence-level adjustments
        adjusted_prob     = self._apply_metric_specific_adjustments(metric_name     = metric_name,
                                                                    sentence        = sentence,
                                                                    base_prob       = base_prob,
                                                                    sentence_length = sentence_length,
                                                                    thresholds      = metric_thresholds,
                                                                   )
        
        return adjusted_prob
    

    def _create_sentence_metric_result(self, metric_name: str, synthetic_prob: float, doc_result: MetricResult, sentence_length: int) -> MetricResult:
        """
        Create sentence-level MetricResult from document-level result
        """
        # Calculate confidence based on sentence characteristics
        sentence_confidence = self._calculate_sentence_confidence(doc_confidence  = doc_result.confidence, 
                                                                  sentence_length = sentence_length,
                                                                 ) 
        
        return MetricResult(metric_name           = metric_name,
                            synthetic_probability = synthetic_prob,
                            authentic_probability = 1.0 - synthetic_prob,
                            hybrid_probability    = 0.0,
                            confidence            = sentence_confidence,
                            details               = doc_result.details,
                            error                 = None,
                           )
    

    def _calculate_sentence_confidence(self, doc_confidence: float, sentence_length: int) -> float:
        """
        Calculate confidence for sentence-level analysis with length consideration
        """
        base_reduction = 0.8
        # Scale confidence penalty with sentence length
        length_penalty = max(0.3, min(1.0, sentence_length / 12.0))  # Normalize around 12 words
        
        return max(0.1, doc_confidence * base_reduction * length_penalty)
    

    def _fallback_weighted_probability(self, metric_results: Dict[str, MetricResult], weights: Dict[str, float], breakdown: Dict[str, float]) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Fallback weighted probability calculation
        """
        weighted_synthetic_probs = list()
        weighted_authentic_probs = list()
        confidences              = list()
        total_weight             = 0.0
        
        for name, result in metric_results.items():
            if result.error is None:
                weight = weights.get(name, 0.0)
                
                if (weight > 0):
                    weighted_synthetic_probs.append(result.synthetic_probability * weight)
                    weighted_authentic_probs.append(result.authentic_probability * weight)
                    confidences.append(result.confidence)
                    total_weight += weight
        
        if not weighted_synthetic_probs or total_weight == 0:
            return 0.5, 0.5, 0.0, 0.5, breakdown or {}
        
        synthetic_prob = sum(weighted_synthetic_probs) / total_weight
        authentic_prob = sum(weighted_authentic_probs) / total_weight
        hybrid_prob    = 0.0  # Fallback
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return synthetic_prob, authentic_prob, hybrid_prob, avg_confidence, breakdown
    

    def _get_document_ensemble_probability(self, ensemble_result: Optional[EnsembleResult], metric_results: Dict[str, MetricResult], weights: Dict[str, float]) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Get document-level ensemble probability
        """
        if ensemble_result:
            # Use existing ensemble result
            breakdown = {name: result.synthetic_probability for name, result in metric_results.items()}
            return (ensemble_result.synthetic_probability,
                    ensemble_result.authentic_probability,
                    ensemble_result.hybrid_probability,
                    ensemble_result.overall_confidence,
                    breakdown
                   )

        else:
            # Calculate from metrics
            return self._fallback_weighted_probability(metric_results, weights, {})


    def _apply_domain_specific_adjustments(self, sentence: str, synthetic_prob: float, sentence_length: int) -> float:
        """
        Apply domain-specific adjustments to Synthetic probability with limits
        """
        original_prob  = synthetic_prob
        adjustments    = list()
        sentence_lower = sentence.lower()
        
        # Technical & AI/ML domains
        if self.domain in [Domain.AI_ML, Domain.SOFTWARE_DEV, Domain.TECHNICAL_DOC, Domain.ENGINEERING, Domain.SCIENCE]:
            if self._has_technical_terms(sentence_lower):
                adjustments.append(1.1)
            
            elif self._has_code_like_patterns(sentence):
                adjustments.append(1.15)
            
            elif (sentence_length > 35):
                adjustments.append(1.05)
                
        # Creative & informal domains
        elif self.domain in [Domain.CREATIVE, Domain.SOCIAL_MEDIA, Domain.BLOG_PERSONAL]:
            if self._has_informal_language(sentence_lower):
                adjustments.append(0.7)

            elif self._has_emotional_language(sentence):
                adjustments.append(0.8)
            
            elif sentence_length < 10:
                adjustments.append(0.8)
                
        # Academic & formal domains
        elif self.domain in [Domain.ACADEMIC, Domain.LEGAL, Domain.MEDICAL]:
            if self._has_citation_patterns(sentence):
                adjustments.append(0.8)
            
            elif self._has_technical_terms(sentence_lower):
                adjustments.append(1.1)
            
            elif (sentence_length > 40):
                adjustments.append(1.1)
                
        # Business & professional domains
        elif self.domain in [Domain.BUSINESS, Domain.MARKETING, Domain.JOURNALISM]:
            if self._has_business_jargon(sentence_lower):
                adjustments.append(1.05)
            
            elif self._has_ambiguous_phrasing(sentence_lower):
                adjustments.append(0.9)
            
            elif (15 <= sentence_length <= 25):
                adjustments.append(0.9)
                
        # Tutorial & educational domains
        elif (self.domain == Domain.TUTORIAL):
            if self._has_instructional_language(sentence_lower):
                adjustments.append(0.85)
            
            elif self._has_step_by_step_pattern(sentence):
                adjustments.append(0.8)
            
            elif self._has_examples(sentence):
                adjustments.append(0.9)
        
        # General domain - minimal adjustments
        elif (self.domain == Domain.GENERAL):
            if self._has_complex_structure(sentence):
                adjustments.append(0.9)
            
            elif self._has_repetition(sentence):
                adjustments.append(1.1)
        
        # Apply adjustments with limits - take strongest 2 adjustments maximum
        if adjustments:
            # Sort by impact (farthest from 1.0)
            adjustments.sort(key = lambda x: abs(x - 1.0), reverse = True)
            
            # Limit to 2 strongest
            strongest_adjustments = adjustments[:2]
            
            for adjustment in strongest_adjustments:
                synthetic_prob *= adjustment
        
        # Ensure probability stays within bounds and doesn't change too drastically
        max_change   = 0.3  # Maximum 30% change from original
        bounded_prob = max(original_prob - max_change, min(original_prob + max_change, synthetic_prob))
        
        return max(0.0, min(1.0, bounded_prob))
    

    def _apply_metric_specific_adjustments(self, metric_name: str, sentence: str, base_prob: float, sentence_length: int, thresholds: MetricThresholds) -> float:
        """
        Apply metric-specific adjustments
        """
        # Use metrics from ensemble
        if (metric_name == "perplexity"):
            if (sentence_length < 8):
                return min(1.0, base_prob * 1.2)
            
            elif (sentence_length > 25):
                return max(0.0, base_prob * 0.8)
        
        elif (metric_name == "entropy"):
            words = sentence.split()
            
            if (len(words) > 3):
                unique_words = len(set(words))
                diversity    = unique_words / len(words)

                if (diversity < 0.6):
                    return min(1.0, base_prob * 1.2)

                elif (diversity > 0.8):
                    return max(0.0, base_prob * 0.8)
        
        elif (metric_name == "linguistic"):
            complexity_score = self._analyze_sentence_complexity(sentence)

            if (complexity_score < 0.3):
                return min(1.0, base_prob * 1.1)
            
            elif (complexity_score > 0.7):
                return max(0.0, base_prob * 0.9)
        
        elif (metric_name == "structural"):
            if ((sentence_length < 5) or (sentence_length > 40)):
                return max(0.0, base_prob * 0.8)

            elif (8 <= sentence_length <= 20):
                return min(1.0, base_prob * 1.1)
        
        elif (metric_name == "semantic_analysis"):
            if self._has_repetition(sentence):
                return min(1.0, base_prob * 1.2)
        
        elif (metric_name == "multi_perturbation_stability"):
            # MultiPerturbationStability adjustments for sentence level
            if (sentence_length > 15):
                return min(1.0, base_prob * 1.1)
        
        return base_prob
    

    def _is_hybrid_content(self, synthetic_prob: float, hybrid_prob: float, confidence: float) -> bool:
        """
        Determine if content should be classified as hybrid
        """
        # Case 1: Explicit high hybrid probability from ensemble
        if (hybrid_prob > self.HYBRID_PROB_THRESHOLD):
            return True
        
        # Case 2: High uncertainty combined with ambiguous synthetic probability
        if (confidence < 0.3 and 0.4 <= synthetic_prob <= 0.7):
            return True
        
        # Case 3: Synthetic probability in hybrid range (0.6-0.8)
        if (0.6 <= synthetic_prob < 0.8):
            return True
        
        return False
    

    def _get_color_for_probability(self, synthetic_prob: float, is_hybrid_content: bool = False, hybrid_prob: float = 0.0) -> Tuple[str, str, str]:
        """
        Get color class with simplified 4-category system
        """
        # Handle hybrid content first
        if is_hybrid_content:
            return "hybrid", "#e9d5ff", f"Mixed synthetic/authentic content ({hybrid_prob:.1%} hybrid)"
        
        # Iterate through simplified thresholds
        for min_thresh, max_thresh, color_class, color_hex, tooltip in self.COLOR_THRESHOLDS:
            if (min_thresh <= synthetic_prob < max_thresh):
                return color_class, color_hex, tooltip
        
        # Fallback for edge cases
        return "uncertain", "#fef3c7", "Uncertain authorship"
    

    def _generate_ensemble_tooltip(self, sentence: str, synthetic_prob: float, authentic_prob: float, hybrid_prob: float, confidence: float, confidence_level: ConfidenceLevel, 
                                   tooltip_base: str, breakdown: Optional[Dict[str, float]] = None, is_hybrid_content: bool = False) -> str:
        """
        Generate enhanced tooltip with ENSEMBLE information
        """
        tooltip = f"{tooltip_base}\n"
        
        if is_hybrid_content:
            tooltip += "🔀 HYBRID CONTENT DETECTED\n"
        
        tooltip += f"Synthetic Probability: {synthetic_prob:.1%}\n"
        tooltip += f"Authentic Probability: {authentic_prob:.1%}\n"
        tooltip += f"Hybrid Probability: {hybrid_prob:.1%}\n"
        tooltip += f"Confidence: {confidence:.1%} ({confidence_level.value.replace('_', ' ').title()})\n"
        tooltip += f"Domain: {self.domain.value.replace('_', ' ').title()}\n"
        tooltip += f"Length: {len(sentence.split())} words"
        
        if breakdown:
            tooltip += "\n\nMetric Breakdown:"
            # Show top 4 metrics
            for metric, prob in list(breakdown.items())[:4]:
                tooltip += f"\n• {metric}: {prob:.1%}"
        
        tooltip += f"\n\nEnsemble Method: {getattr(self.ensemble, 'primary_method', 'fallback')}"
        
        return tooltip


    def _has_citation_patterns(self, sentence: str) -> bool:
        """
        Check for academic citation patterns
        """
        citation_indicators = ['et al.', 'ibid.', 'cf.', 'e.g.', 'i.e.', 'vol.', 'pp.', 'ed.', 'trans.', 'reference', 'cited', 'according to']
        return any(indicator in sentence.lower() for indicator in citation_indicators)
    

    def _has_informal_language(self, sentence: str) -> bool:
        """
        Check for informal language patterns
        """
        informal_indicators = ['lol', 'omg', 'btw', 'imo', 'tbh', 'afaik', 'smh', '👋', '😂', '❤️', 'haha', 'wow', 'awesome']
        return any(indicator in sentence.lower() for indicator in informal_indicators)
    

    def _has_technical_terms(self, sentence: str) -> bool:
        """
        Check for domain-specific technical terms
        """
        technical_indicators = ['hereinafter', 'whereas', 'aforementioned', 'diagnosis', 'prognosis', 'etiology',
                                'algorithm', 'neural network', 'machine learning', 'api', 'endpoint', 'database',
                                'quantum', 'thermodynamics', 'hypothesis', 'methodology']

        return any(indicator in sentence.lower() for indicator in technical_indicators)
    

    def _has_ambiguous_phrasing(self, sentence: str) -> bool:
        """
        Check for ambiguous phrasing that might indicate human writing
        """
        ambiguous_indicators = ['perhaps', 'maybe', 'possibly', 'likely', 'appears to', 'seems to', 'might be', 'could be']
        return any(indicator in sentence.lower() for indicator in ambiguous_indicators)
    

    def _has_complex_structure(self, sentence: str) -> bool:
        """
        Check if sentence has complex linguistic structure
        """
        words = sentence.split()
        if (len(words) < 8):
            return False
        
        complex_indicators = ['which', 'that', 'although', 'because', 'while', 'when', 'if', 'however', 'therefore']
        return any(indicator in sentence.lower() for indicator in complex_indicators)
    

    def _has_emotional_language(self, sentence: str) -> bool:
        """
        Check for emotional or subjective language
        """
        emotional_indicators = ['feel', 'believe', 'think', 'wonder', 'hope', 'wish', 'love', 'hate', 'frustrating', 'exciting']
        return any(indicator in sentence.lower() for indicator in emotional_indicators)
    

    def _has_business_jargon(self, sentence: str) -> bool:
        """
        Check for business jargon
        """
        jargon_indicators = ['synergy', 'leverage', 'bandwidth', 'circle back', 'touch base', 'value add', 'core competency']
        return any(indicator in sentence.lower() for indicator in jargon_indicators)
    

    def _has_instructional_language(self, sentence: str) -> bool:
        """
        Check for instructional language patterns
        """
        instructional_indicators = ['step by step', 'firstly', 'secondly', 'finally', 'note that', 'remember to', 'make sure']
        return any(indicator in sentence.lower() for indicator in instructional_indicators)
    

    def _has_step_by_step_pattern(self, sentence: str) -> bool:
        """
        Check for step-by-step instructions
        """
        step_patterns = ['step 1', 'step 2', 'step 3', 'step one', 'step two', 'first step', 'next step']
        return any(pattern in sentence.lower() for pattern in step_patterns)
    

    def _has_examples(self, sentence: str) -> bool:
        """
        Check for example indicators
        """
        example_indicators = ['for example', 'for instance', 'such as', 'e.g.', 'as an example']
        return any(indicator in sentence.lower() for indicator in example_indicators)
    

    def _has_code_like_patterns(self, sentence: str) -> bool:
        """
        Check for code-like patterns in technical domains
        """
        code_patterns = ['function', 'variable', 'class', 'method', 'import', 'def ', 'void ', 'public ', 'private ']
        return any(pattern in sentence for pattern in code_patterns)
    

    def _analyze_sentence_complexity(self, sentence: str) -> float:
        """
        Analyze sentence complexity (0 = simple, 1 = complex)
        """
        words = sentence.split()
        if (len(words) < 5):
            return 0.2
        
        complexity_indicators = ['although', 'because', 'while', 'when', 'if', 'since', 'unless', 'until', 'which', 'that', 'who', 'whom', 'whose', 'and', 'but', 'or', 'yet', 'so', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', ',', ';', ':', '—']
        
        score                 = 0.0
        
        if (len(words) > 15):
            score += 0.3

        elif (len(words) > 25):
            score += 0.5
        
        indicator_count       = sum(1 for indicator in complexity_indicators if indicator in sentence.lower())
        score                += min(0.5, indicator_count * 0.1)
        
        clause_indicators     = [',', ';', 'and', 'but', 'or', 'because', 'although']
        clause_count          = sum(1 for indicator in clause_indicators if indicator in sentence.lower())
        score                += min(0.2, clause_count * 0.05)
        
        return min(1.0, score)
    

    def _has_repetition(self, sentence: str) -> bool:
        """
        Check if sentence has word repetition (common in Synthetic text)
        """
        words = sentence.lower().split()
        if (len(words) < 6):
            return False
        
        word_counts = dict()

        for word in words:
            if (len(word) > 3):
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 2]
        return (len(repeated_words) > 0)
    

    def generate_html(self, highlighted_sentences: List[HighlightedSentenceResult], include_legend: bool = True) -> str:
        """
        Generate HTML with highlighted sentences
        
        Arguments:
        ----------
            highlighted_sentences { List[HighlightedSentenceResult] } : Sentences with highlighting data

            include_legend                   { bool }                 : Whether to include legend
            
        Returns:
        --------
                                 { str }                              : HTML content
        """
        html_parts = list()
        
        # Add CSS
        html_parts.append(self._generate_css())
        
        # Include legend if requested
        if include_legend:
            html_parts.append(self._generate_legend_html())
        
        # Add highlighted text container
        html_parts.append('<div class="highlighted-text">')
        
        for sent in highlighted_sentences:
            extra_class = " hybrid-highlight" if sent.is_hybrid_content else ""
            html_parts.append(f'<span class="highlight {sent.color_class}{extra_class}" '
                              f'data-synthetic-prob="{sent.synthetic_probability:.4f}" '
                              f'data-authentic-prob="{sent.authentic_probability:.4f}" '
                              f'data-hybrid-prob="{sent.hybrid_probability:.4f}" '
                              f'data-confidence="{sent.confidence:.4f}" '
                              f'data-confidence-level="{sent.confidence_level.value}" '
                              f'data-domain="{self.domain.value}" '
                              f'data-sentence-idx="{sent.index}" '
                              f'data-is-hybrid="{str(sent.is_hybrid_content).lower()}" '
                              f'title="{sent.tooltip}">'
                              f'{sent.text}'
                              f'</span> ')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
        

    def _generate_css(self) -> str:
        """
        Generate CSS for highlighting for better readability with 4 color types
        """
        return """
        <style>
        .highlighted-text {
            line-height: 1.8;
            font-size: 16px;
            font-family: 'Georgia', serif;
            padding: 20px;
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .highlight {
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 3px;
            transition: all 0.2s ease;
            cursor: help;
            border-bottom: 2px solid transparent;
            color: #000000 !important;
            font-weight: 500;
            position: relative;
        }
        
        .highlight:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10;
            text-shadow: 0 1px 1px rgba(255,255,255,0.8);
        }
        
        /* Authentic - Green tones */
        .authentic {
            background-color: #d1fae5;
            border-bottom-color: #10b981;
        }
        
        /* Uncertain - Yellow tones */
        .uncertain {
            background-color: #fef3c7;
            border-bottom-color: #f59e0b;
        }
        
        /* Hybrid - Purple tones */
        .hybrid {
            background-color: #e9d5ff;
            border-bottom-color: #a855f7;
        }
        
        .hybrid-highlight:hover {
            border: 2px dashed #a855f7;
        }
        
        /* Synthetic - Red tones */
        .synthetic {
            background-color: #fee2e2;
            border-bottom-color: #ef4444;
        }
        </style>
        """
    

    def _generate_legend_html(self) -> str:
        """
        Generate legend HTML for 4-category system
        """
        return """
        <div class="highlight-legend" style="margin-bottom: 20px; padding: 15px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;">
            <h4 style="margin: 0 0 10px 0; font-size: 14px; font-weight: 600; color: #374151;">Text Analysis Legend</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 16px; height: 16px; background: #d1fae5; border: 1px solid #10b981; border-radius: 3px;"></div>
                    <span style="font-size: 12px; color: #374151;">Authentic (0-40% synthetic)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 16px; height: 16px; background: #fef3c7; border: 1px solid #f59e0b; border-radius: 3px;"></div>
                    <span style="font-size: 12px; color: #374151;">Uncertain (40-60% synthetic)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 16px; height: 16px; background: #e9d5ff; border: 1px solid #a855f7; border-radius: 3px;"></div>
                    <span style="font-size: 12px; color: #374151;">Hybrid (60-80% synthetic)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 16px; height: 16px; background: #fee2e2; border: 1px solid #ef4444; border-radius: 3px;"></div>
                    <span style="font-size: 12px; color: #374151;">Synthetic (80-100% synthetic)</span>
                </div>
            </div>
        </div>
        """


# Export
__all__ = ["TextHighlighter"]