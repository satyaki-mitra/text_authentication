# DEPENDENCIES
import re
from typing import List
from typing import Dict
from typing import Tuple
from loguru import logger
from typing import Optional
from dataclasses import dataclass
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from detector.ensemble import EnsembleResult
from detector.ensemble import EnsembleClassifier
from processors.text_processor import TextProcessor
from config.threshold_config import ConfidenceLevel
from config.threshold_config import MetricThresholds
from config.threshold_config import get_confidence_level
from config.threshold_config import get_threshold_for_domain
from config.threshold_config import get_active_metric_weights


@dataclass
class HighlightedSentence:
    """
    A sentence with highlighting information - ENHANCED FOR ENSEMBLE INTEGRATION
    """
    text              : str
    ai_probability    : float
    human_probability : float
    mixed_probability : float
    confidence        : float
    confidence_level  : ConfidenceLevel
    color_class       : str
    tooltip           : str
    index             : int
    is_mixed_content  : bool
    metric_breakdown  : Optional[Dict[str, float]] = None


class TextHighlighter:
    """
    Generates sentence-level highlighting with ensemble resaults integration
    
    FEATURES:
    - Sentence-level highlighting with confidence scores
    - Domain-aware calibration
    - Ensemble-based probability aggregation
    - Mixed content detection
    - Explainable tooltips
    """
    # Color thresholds with MIXED content support
    COLOR_THRESHOLDS = [(0.00, 0.10, "very-high-human", "#dcfce7", "Very likely human-written"),
                        (0.10, 0.25, "high-human", "#bbf7d0", "Likely human-written"),
                        (0.25, 0.40, "medium-human", "#86efac", "Possibly human-written"),
                        (0.40, 0.60, "uncertain", "#fef9c3", "Uncertain"),
                        (0.60, 0.75, "medium-ai", "#fef3c7", "Possibly AI-generated"),
                        (0.75, 0.90, "high-ai", "#fed7aa", "Likely AI-generated"),
                        (0.90, 1.01, "very-high-ai", "#fecaca", "Very likely AI-generated"),
                       ]
    
    # Mixed content pattern
    MIXED_THRESHOLD = 0.25


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
        self.ensemble           = ensemble_classifier or EnsembleClassifier(primary_method  = "confidence_calibrated",
                                                                            fallback_method = "domain_weighted",
                                                                           )
    

    def generate_highlights(self, text: str, metric_results: Dict[str, MetricResult], ensemble_result: Optional[EnsembleResult] = None,
                            enabled_metrics: Optional[Dict[str, bool]] = None, use_sentence_level: bool = True) -> List[HighlightedSentence]:
        """
        Generate sentence-level highlights with ensemble integration
        
        Arguments:
        ----------
            text                    { str }       : Original text
            
            metric_results          { dict }      : Results from all 6 metrics
            
            ensemble_result    { EnsembleResult } : Optional document-level ensemble result
            
            enabled_metrics         { dict }      : Dict of metric_name -> is_enabled
            
            use_sentence_level      { bool }      : Whether to compute sentence-level probabilities
            
        Returns:
        --------
                         { list }                 : List of HighlightedSentence objects
        """
        # Get domain-appropriate weights for enabled metrics
        if enabled_metrics is None:
            enabled_metrics = {name: True for name in metric_results.keys()}
        
        weights   = get_active_metric_weights(self.domain, enabled_metrics)
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Calculate probabilities for each sentence using ENSEMBLE METHODS
        highlighted_sentences = list()
        
        for idx, sentence in enumerate(sentences):
            if use_sentence_level:
                # Use ENSEMBLE for sentence-level analysis
                ai_prob, human_prob, mixed_prob, confidence, breakdown = self._calculate_sentence_ensemble_probability(sentence        = sentence, 
                                                                                                                       metric_results  = metric_results,
                                                                                                                       weights         = weights,
                                                                                                                       ensemble_result = ensemble_result,
                                                                                                                      )
            else:
                # Use document-level ensemble probabilities
                ai_prob, human_prob, mixed_prob, confidence, breakdown = self._get_document_ensemble_probability(ensemble_result = ensemble_result,
                                                                                                                 metric_results  = metric_results,
                                                                                                                 weights         = weights,
                                                                                                                )
            
            # Determine if this is mixed content
            is_mixed_content                     = (mixed_prob > self.MIXED_THRESHOLD)
            
            # Get confidence level
            confidence_level                     = get_confidence_level(confidence)
            
            # Get color class (consider mixed content)
            color_class, color_hex, tooltip_base = self._get_color_for_probability(probability      = ai_prob,
                                                                                   is_mixed_content = is_mixed_content,
                                                                                   mixed_prob       = mixed_prob,
                                                                                  )
            
            # Generate enhanced tooltip
            tooltip                              = self._generate_ensemble_tooltip(sentence         = sentence, 
                                                                                   ai_prob          = ai_prob,
                                                                                   human_prob       = human_prob,
                                                                                   mixed_prob       = mixed_prob,
                                                                                   confidence       = confidence, 
                                                                                   confidence_level = confidence_level, 
                                                                                   tooltip_base     = tooltip_base, 
                                                                                   breakdown        = breakdown,
                                                                                   is_mixed_content = is_mixed_content,
                                                                                  )
            
            highlighted_sentences.append(HighlightedSentence(text              = sentence,
                                                             ai_probability    = ai_prob,
                                                             human_probability = human_prob,
                                                             mixed_probability = mixed_prob,
                                                             confidence        = confidence,
                                                             confidence_level  = confidence_level,
                                                             color_class       = color_class,
                                                             tooltip           = tooltip,
                                                             index             = idx,
                                                             is_mixed_content  = is_mixed_content,
                                                             metric_breakdown  = breakdown,
                                                            )
                                        )
        
        return highlighted_sentences

    
    def _calculate_sentence_ensemble_probability(self, sentence: str, metric_results: Dict[str, MetricResult], weights: Dict[str, float], 
                                                 ensemble_result: Optional[EnsembleResult] = None) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Calculate sentence probabilities using ensemble methods with domain calibration
        """
        sentence_length = len(sentence.split())
        
        # Skip very short sentences from detailed ensemble analysis
        if (sentence_length < 3):
            return 0.4, 0.5, 0.1, 0.6, {"short_sentence": 0.4}
        
        # Calculate sentence-level metric results
        sentence_metric_results = dict()
        breakdown               = dict()
        
        for name, doc_result in metric_results.items():
            if doc_result.error is None:
                # Compute sentence-level probability for this metric
                sentence_prob                 = self._compute_sentence_metric(metric_name = name,
                                                                              sentence    = sentence,
                                                                              result      = doc_result,
                                                                              weight      = weights.get(name, 0.0),
                                                                             )
                
                # Create sentence-level MetricResult
                sentence_metric_results[name] = self._create_sentence_metric_result(metric_name = name,
                                                                                    ai_prob     = sentence_prob,
                                                                                    doc_result  = doc_result,
                                                                                   )
                
                breakdown[name]               = sentence_prob
        
        # Use ensemble to combine sentence-level metrics
        if sentence_metric_results:
            try:
                ensemble_sentence_result = self.ensemble.predict(metric_results = sentence_metric_results,
                                                                 domain         = self.domain,
                                                                )
                
                return (ensemble_sentence_result.ai_probability, ensemble_sentence_result.human_probability, ensemble_sentence_result.mixed_probability, 
                        ensemble_sentence_result.overall_confidence, breakdown)
                        
            except Exception as e:
                logger.warning(f"Sentence ensemble failed: {e}")
        
        # Fallback: weighted average
        return self._calculate_weighted_probability(metric_results, weights, breakdown)
    

    def _compute_sentence_metric(self, metric_name: str, sentence: str, result: MetricResult, weight: float) -> float:
        """
        Compute metric probability for a single sentence using domain-specific thresholds
        """
        sentence_length   = len(sentence.split())
        
        # Get domain-specific threshold for this metric
        metric_thresholds = getattr(self.domain_thresholds, metric_name, None)

        if not metric_thresholds:
            return result.ai_probability
        
        # Base probability from document-level result
        base_prob         = result.ai_probability
        
        # Apply domain-aware sentence-level adjustments
        adjusted_prob     = self._apply_metric_specific_adjustments(metric_name     = metric_name, 
                                                                    sentence        = sentence, 
                                                                    base_prob       = base_prob, 
                                                                    sentence_length = sentence_length, 
                                                                    thresholds      = metric_thresholds,
                                                                   )
        
        return adjusted_prob
    

    def _create_sentence_metric_result(self, metric_name: str, ai_prob: float, doc_result: MetricResult) -> MetricResult:
        """
        Create sentence-level MetricResult from document-level result
        """
        # Adjust confidence based on sentence characteristics
        sentence_confidence = self._calculate_sentence_confidence(doc_result.confidence)
        
        return MetricResult(metric_name       = metric_name,
                            ai_probability    = ai_prob,
                            human_probability = 1.0 - ai_prob,  
                            mixed_probability = 0.0,  
                            confidence        = sentence_confidence,
                            details           = doc_result.details,
                            error             = None,
                           )
    

    def _calculate_sentence_confidence(self, doc_confidence: float) -> float:
        """
        Calculate confidence for sentence-level analysis
        """
        # Sentence-level analysis typically has lower confidence
        return max(0.1, doc_confidence * 0.8)
    

    def _calculate_weighted_probability(self, metric_results: Dict[str, MetricResult], weights: Dict[str, float], breakdown: Dict[str, float]) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Fallback weighted probability calculation
        """
        weighted_ai_probs    = list()
        weighted_human_probs = list()
        confidences          = list()
        
        for name, result in metric_results.items():
            if (result.error is None):
                weight = weights.get(name, 0.0)
                
                if (weight > 0):
                    weighted_ai_probs.append(result.ai_probability * weight)
                    weighted_human_probs.append(result.human_probability * weight)
                    confidences.append(result.confidence)
        
        if not weighted_ai_probs:
            return 0.5, 0.5, 0.0, 0.0, {}
        
        total_weight   = sum(weights.values())
        ai_prob        = sum(weighted_ai_probs) / total_weight if total_weight > 0 else 0.5
        human_prob     = sum(weighted_human_probs) / total_weight if total_weight > 0 else 0.5
        mixed_prob     = 0.0  # Fallback
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ai_prob, human_prob, mixed_prob, avg_confidence, breakdown
    

    def _get_document_ensemble_probability(self, ensemble_result: Optional[EnsembleResult], metric_results: Dict[str, MetricResult],
                                           weights: Dict[str, float]) -> Tuple[float, float, float, float, Dict[str, float]]:
        """
        Get document-level ensemble probability
        """
        if ensemble_result:
            # Use existing ensemble result
            breakdown = {name: result.ai_probability for name, result in metric_results.items()}
            return (ensemble_result.ai_probability, ensemble_result.human_probability, ensemble_result.mixed_probability,
                    ensemble_result.overall_confidence, breakdown)
        
        else:
            # Calculate from metrics
            return self._calculate_weighted_probability(metric_results, weights, {})
    

    def _apply_domain_specific_adjustments(self, sentence: str, ai_prob: float, sentence_length: int) -> float:
        """
        Apply domain-specific adjustments to AI probability - ENHANCED
        """
        # Your existing domain adjustment logic is good, keeping it
        if (self.domain == Domain.CREATIVE):
            if (sentence_length > 30):
                ai_prob *= 0.9

            elif (self._has_complex_structure(sentence)):
                ai_prob *= 0.85
                
        elif (self.domain == Domain.ACADEMIC):
            if (sentence_length > 40):
                ai_prob *= 1.1

            elif (self._has_citation_patterns(sentence)):
                ai_prob *= 0.8
                
        elif (self.domain == Domain.SOCIAL_MEDIA):
            if (sentence_length < 10):
                ai_prob *= 0.7

            elif (self._has_informal_language(sentence)):
                ai_prob *= 0.8
                
        elif (self.domain in [Domain.LEGAL, Domain.MEDICAL]):
            if (self._has_technical_terms(sentence)):
                ai_prob *= 1.1

            elif (self._has_ambiguous_phrasing(sentence)):
                ai_prob *= 0.9
        
        return max(0.0, min(1.0, ai_prob))
    

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
        
        elif (metric_name == "detect_gpt"):
            # DetectGPT adjustments for sentence level
            if (sentence_length > 15):
                return min(1.0, base_prob * 1.1)
        
        return base_prob
    

    def _get_color_for_probability(self, probability: float, is_mixed_content: bool = False, mixed_prob: float = 0.0) -> Tuple[str, str, str]:
        """
        Get color class with mixed content support
        """
        if is_mixed_content and mixed_prob > self.MIXED_THRESHOLD:
            return "mixed-content", "#e9d5ff", f"Mixed AI/Human content ({mixed_prob:.1%} mixed)"
        
        for min_thresh, max_thresh, color_class, color_hex, tooltip in self.COLOR_THRESHOLDS:
            if (min_thresh <= probability < max_thresh):
                return color_class, color_hex, tooltip
        
        return "uncertain", "#fef9c3", "Uncertain"
    

    def _generate_ensemble_tooltip(self, sentence: str, ai_prob: float, human_prob: float, mixed_prob: float, confidence: float, confidence_level: ConfidenceLevel, 
                                  tooltip_base: str, breakdown: Optional[Dict[str, float]] = None, is_mixed_content: bool = False) -> str:
        """
        Generate enhanced tooltip with ENSEMBLE information
        """
        tooltip = f"{tooltip_base}\n"
        
        if is_mixed_content:
            tooltip += "🔀 MIXED CONTENT DETECTED\n"
        
        tooltip += f"AI Probability: {ai_prob:.1%}\n"
        tooltip += f"Human Probability: {human_prob:.1%}\n"
        tooltip += f"Mixed Probability: {mixed_prob:.1%}\n"
        tooltip += f"Confidence: {confidence:.1%} ({confidence_level.value.replace('_', ' ').title()})\n"
        tooltip += f"Domain: {self.domain.value.title()}\n"
        tooltip += f"Length: {len(sentence.split())} words"
        
        if breakdown:
            tooltip += "\n\nMetric Breakdown:"
            # Show top 4 metrics
            for metric, prob in list(breakdown.items())[:4]:  
                tooltip += f"\n• {metric}: {prob:.1%}"
        
        tooltip += f"\n\nEnsemble Method: {self.ensemble.primary_method}"
        
        return tooltip


    def _has_citation_patterns(self, sentence: str) -> bool:
        """
        Check for academic citation patterns
        """
        citation_indicators = ['et al.', 'ibid.', 'cf.', 'e.g.', 'i.e.', 'vol.', 'pp.', 'ed.', 'trans.']
        return any(indicator in sentence for indicator in citation_indicators)
    

    def _has_informal_language(self, sentence: str) -> bool:
        """
        Check for informal language patterns
        """
        informal_indicators = ['lol', 'omg', 'btw', 'imo', 'tbh', 'afaik', 'smh', '👋', '😂', '❤️']
        return any(indicator in sentence.lower() for indicator in informal_indicators)
    

    def _has_technical_terms(self, sentence: str) -> bool:
        """
        Check for domain-specific technical terms
        """
        technical_indicators = ['hereinafter', 'whereas', 'aforementioned', 'diagnosis', 'prognosis', 'etiology']
        return any(indicator in sentence.lower() for indicator in technical_indicators)
    

    def _has_ambiguous_phrasing(self, sentence: str) -> bool:
        """
        Check for ambiguous phrasing that might indicate human writing
        """
        ambiguous_indicators = ['perhaps', 'maybe', 'possibly', 'likely', 'appears to', 'seems to']
        return any(indicator in sentence.lower() for indicator in ambiguous_indicators)
    

    def _has_complex_structure(self, sentence: str) -> bool:
        """
        Check if sentence has complex linguistic structure
        """
        words = sentence.split()
        if len(words) < 8:
            return False
        complex_indicators = ['which', 'that', 'although', 'because', 'while', 'when', 'if']
        return any(indicator in sentence.lower() for indicator in complex_indicators)
    

    def _analyze_sentence_complexity(self, sentence: str) -> float:
        """
        Analyze sentence complexity (0 = simple, 1 = complex)
        """
        words = sentence.split()
        if len(words) < 5:
            return 0.2
        
        complexity_indicators = ['although', 'because', 'while', 'when', 'if', 'since', 'unless', 'until', 'which', 'that', 'who', 'whom', 'whose', 'and', 'but', 'or', 'yet', 'so', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless', ',', ';', ':', '—']
        
        score                 = 0.0
        
        if (len(words) > 15):
            score += 0.3

        elif (len(words) > 25):
            score += 0.5
        
        indicator_count   = sum(1 for indicator in complexity_indicators if indicator in sentence.lower())
        score            += min(0.5, indicator_count * 0.1)
        
        clause_indicators = [',', ';', 'and', 'but', 'or', 'because', 'although']
        clause_count      = sum(1 for indicator in clause_indicators if indicator in sentence.lower())
        score             += min(0.2, clause_count * 0.05)
        
        return min(1.0, score)
    

    def _has_repetition(self, sentence: str) -> bool:
        """
        Check if sentence has word repetition (common in AI text)
        """
        words = sentence.lower().split()
        if len(words) < 6:
            return False
        
        word_counts = dict()

        for word in words:
            if (len(word) > 3):
                word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 2]
        
        return len(repeated_words) > 0
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split the text chunk into multiple sentences
        """
        sentences          = self.text_processor.split_sentences(text)
        filtered_sentences = list()

        for sentence in sentences:
            clean_sentence = sentence.strip()
            if (len(clean_sentence) >= 10):
                filtered_sentences.append(clean_sentence)
        
        return filtered_sentences


    def generate_html(self, highlighted_sentences: List[HighlightedSentence], include_legend: bool = True, include_metrics: bool = False) -> str:
        """
        Generate HTML with highlighted sentences
        """
        html_parts = list()
        
        # Add CSS
        html_parts.append(self._generate_enhanced_css())
        
        # Add legend if requested
        if include_legend:
            html_parts.append(self._generate_legend_html())
        
        # Add highlighted text container
        html_parts.append('<div class="highlighted-text">')
        
        for sent in highlighted_sentences:
            extra_class = " mixed-highlight" if sent.is_mixed_content else ""
            html_parts.append(f'<span class="highlight {sent.color_class}{extra_class}" '
                            f'data-ai-prob="{sent.ai_probability:.4f}" '
                            f'data-human-prob="{sent.human_probability:.4f}" '
                            f'data-mixed-prob="{sent.mixed_probability:.4f}" '
                            f'data-confidence="{sent.confidence:.4f}" '
                            f'data-confidence-level="{sent.confidence_level.value}" '
                            f'data-domain="{self.domain.value}" '
                            f'data-sentence-idx="{sent.index}" '
                            f'data-is-mixed="{str(sent.is_mixed_content).lower()}" '
                            f'title="{sent.tooltip}">'
                            f'{sent.text}'
                            f'</span> ')
        
        html_parts.append('</div>')
        
        # Add metrics summary if requested
        if include_metrics and highlighted_sentences:
            html_parts.append(self._generate_metrics_summary(highlighted_sentences))
        
        return '\n'.join(html_parts)
    

    def _generate_enhanced_css(self) -> str:
        """
        Generate CSS for highlighting
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
        }
        
        .highlight:hover {
            transform: scale(1.02);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            z-index: 10;
            position: relative;
        }
        
        /* AI indicators */
        .very-high-ai {
            background-color: #fecaca;
            border-bottom-color: #ef4444;
        }
        
        .high-ai {
            background-color: #fed7aa;
            border-bottom-color: #f97316;
        }
        
        .medium-ai {
            background-color: #fef3c7;
            border-bottom-color: #f59e0b;
        }
        
        /* Uncertain */
        .uncertain {
            background-color: #fef9c3;
            border-bottom-color: #fbbf24;
        }
        
        /* Human indicators */
        .medium-human {
            background-color: #ecfccb;
            border-bottom-color: #a3e635;
        }
        
        .high-human {
            background-color: #bbf7d0;
            border-bottom-color: #4ade80;
        }
        
        .very-high-human {
            background-color: #dcfce7;
            border-bottom-color: #22c55e;
        }
        
        /* Mixed content */
        .mixed-content {
            background-color: #e9d5ff;
            border-bottom-color: #a855f7;
            background-image: repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(168, 85, 247, 0.1) 5px, rgba(168, 85, 247, 0.1) 10px);
        }
        
        .mixed-highlight:hover {
            border: 2px dashed #a855f7;
        }
        
        /* Legend and summary styles */
        .highlight-legend, .highlight-summary {
            margin-bottom: 20px;
            padding: 15px;
            background: #f9fafb;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        
        .highlight-legend h4, .highlight-summary h4 {
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: 600;
            color: #374151;
        }
        
        .legend-items {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }
        
        .legend-color {
            width: 40px;
            height: 20px;
            border-radius: 4px;
            display: inline-block;
        }
        
        .legend-label {
            color: #6b7280;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }
        
        .stat-label {
            font-size: 13px;
            color: #6b7280;
        }
        
        .stat-value {
            font-size: 13px;
            font-weight: 600;
            color: #374151;
        }
        </style>
        """
    

    def _generate_metrics_summary(self, sentences: List[HighlightedSentence]) -> str:
        """
        Generate summary statistics for highlighted sentences
        """
        if not sentences:
            return ""
        
        ai_probs            = [s.ai_probability for s in sentences]
        avg_ai_prob         = sum(ai_probs) / len(ai_probs)
        
        # Count sentences by category
        ai_sentences        = len([s for s in sentences if s.ai_probability >= 0.6])
        human_sentences     = len([s for s in sentences if s.ai_probability <= 0.4])
        uncertain_sentences = len([s for s in sentences if 0.4 < s.ai_probability < 0.6])
        mixed_sentences     = len([s for s in sentences if s.is_mixed_content])
        
        html                = f"""
        <div class="highlight-summary">
            <h4>Text Analysis Summary</h4>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-label">Average AI Probability</span>
                    <span class="stat-value">{avg_ai_prob:.1%}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">AI-like Sentences</span>
                    <span class="stat-value">{ai_sentences} ({ai_sentences/len(sentences):.1%})</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Human-like Sentences</span>
                    <span class="stat-value">{human_sentences} ({human_sentences/len(sentences):.1%})</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Uncertain Sentences</span>
                    <span class="stat-value">{uncertain_sentences} ({uncertain_sentences/len(sentences):.1%})</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Mixed Content Sentences</span>
                    <span class="stat-value">{mixed_sentences} ({mixed_sentences/len(sentences):.1%})</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Domain</span>
                    <span class="stat-value">{self.domain.value.title()}</span>
                </div>
            </div>
        </div>
        """
        return html
    

    def _generate_legend_html(self) -> str:
        """
        Generate HTML for color legend
        """
        html  = '<div class="highlight-legend">'
        html += '<h4>Detection Legend:</h4>'
        html += '<div class="legend-items">'
        
        # Add mixed content legend item
        html += (f'<div class="legend-item">'
                  f'<span class="legend-color mixed-content"></span>'
                  f'<span class="legend-label">Mixed AI/Human Content</span>'
                  f'</div>'
                 )
        
        for min_t, max_t, color_class, color_hex, label in self.COLOR_THRESHOLDS:
            range_text = f"{min_t:.0%}-{max_t:.0%}" if max_t < 1.01 else f"{min_t:.0%}+"
            html      += (f'<div class="legend-item">'
                          f'<span class="legend-color {color_class}"></span>'
                          f'<span class="legend-label">{label} ({range_text})</span>'
                          f'</div>'
                         )
        
        html += '</div></div>'
        return html


# Export
__all__ = ["TextHighlighter", 
           "HighlightedSentence",
          ]