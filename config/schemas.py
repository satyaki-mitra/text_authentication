# DEPENDENCIES
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import field
from datetime import datetime
from config.enums import Script
from config.enums import Domain
from config.enums import Language
from dataclasses import dataclass
from config.enums import ModelType
from config.enums import ConfidenceLevel


@dataclass
class ModelConfig:
    """
    Configuration for a single model
    """
    model_id          : str
    model_type        : ModelType
    description       : str
    size_mb           : int
    required          : bool           = True
    download_priority : int            = 1     # 1=highest, 5=lowest
    quantizable       : bool           = True
    onnx_compatible   : bool           = False
    cache_model       : bool           = True
    max_length        : Optional[int]  = None
    batch_size        : int            = 1
    additional_params : Dict[str, Any] = field(default_factory = dict)


@dataclass
class ModelUsageStats:
    """
    Lightweight model usage statistics
    """
    model_name               : str
    usage_count              : int
    last_used                : datetime
    timed_usage_count        : int     
    total_usage_time_seconds : float
    avg_usage_time_seconds   : float
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"model_name"               : self.model_name,
                "usage_count"              : self.usage_count,
                "last_used"                : self.last_used.isoformat() if self.last_used else None,
                "timed_usage_count"        : self.timed_usage_count,
                "total_usage_time_seconds" : round(self.total_usage_time_seconds, 2),
                "avg_usage_time_seconds"   : round(self.avg_usage_time_seconds, 2),
               }


@dataclass
class ExtractedDocument:
    """
    Container for extracted document content with metadata
    """
    text              : str
    file_path         : Optional[str]
    file_type         : str
    file_size_bytes   : int
    page_count        : int
    extraction_method : str
    metadata          : Dict[str, Any]
    is_success        : bool
    error_message     : Optional[str]
    warnings          : List[str]
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"text_length"        : len(self.text),
                "file_type"          : self.file_type,
                "file_size_bytes"    : self.file_size_bytes,
                "page_count"         : self.page_count,
                "extraction_method"  : self.extraction_method,
                "metadata"           : self.metadata,
                "is_success"         : self.is_success,
                "error_message"      : self.error_message,
                "warnings"           : self.warnings,
               }



@dataclass
class ProcessedText:
    """
    Container for processed text with metadata
    """
    original_text      : str
    cleaned_text       : str
    sentences          : List[str]
    words              : List[str]
    paragraphs         : List[str]
    char_count         : int
    word_count         : int
    sentence_count     : int
    paragraph_count    : int
    avg_sentence_length: float
    avg_word_length    : float
    is_valid           : bool
    validation_errors  : List[str]
    metadata           : Dict[str, Any]
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"original_length"      : len(self.original_text),
                "cleaned_length"       : len(self.cleaned_text),
                "char_count"           : self.char_count,
                "word_count"           : self.word_count,
                "sentence_count"       : self.sentence_count,
                "paragraph_count"      : self.paragraph_count,
                "avg_sentence_length"  : round(self.avg_sentence_length, 2),
                "avg_word_length"      : round(self.avg_word_length, 2),
                "is_valid"             : self.is_valid,
                "validation_errors"    : self.validation_errors,
                "metadata"             : self.metadata,
               }


@dataclass
class LanguageDetectionResult:
    """
    Result of language detection
    """
    primary_language   : Language
    evidence_strength  : float
    all_languages      : Dict[str, float]  # language_code -> evidence_strength
    script             : Script
    is_multilingual    : bool
    detection_method   : str
    char_count         : int
    word_count         : int
    warnings           : List[str]
    

    def to_dict(self) -> Dict:
        """
        Convert to dictionary
        """
        return {"primary_language"  : self.primary_language.value,
                "evidence_strength" : round(self.evidence_strength, 4),
                "all_languages"     : {k: round(v, 4) for k, v in self.all_languages.items()},
                "script"            : self.script.value,
                "is_multilingual"   : self.is_multilingual,
                "detection_method"  : self.detection_method,
                "char_count"        : self.char_count,
                "word_count"        : self.word_count,
                "warnings"          : self.warnings,
               }


@dataclass
class MetricThresholds:
    """
    Thresholds for a single metric
    """
    synthetic_threshold   : float       # Above this = low authenticity
    authentic_threshold   : float       # Below this = high authenticity
    weight                : float
    confidence_multiplier : float = 1.0
    


@dataclass
class DomainThresholds:
    """
    Thresholds for 6 metrics in a specific domain
    """
    domain                       : Domain
    structural                   : MetricThresholds
    perplexity                   : MetricThresholds
    entropy                      : MetricThresholds
    semantic                     : MetricThresholds
    linguistic                   : MetricThresholds
    multi_perturbation_stability : MetricThresholds
    ensemble_threshold           : float            = 0.5     # authenticity decision boundary


@dataclass
class DomainPrediction:
    """
    Result of domain classification
    """
    primary_domain    : Domain
    secondary_domain  : Optional[Domain]
    evidence_strength : float
    domain_scores     : Dict[str, float]


class MetricResult:
    """
    Result from a metric calculation
    """
    def __init__(self, metric_name: str, synthetic_probability: float, authentic_probability: float, hybrid_probability: float, confidence: float, details: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.metric_name           = metric_name
        self.synthetic_probability = max(0.0, min(1.0, synthetic_probability))
        self.authentic_probability = max(0.0, min(1.0, authentic_probability))
        self.hybrid_probability    = max(0.0, min(1.0, hybrid_probability))
        self.confidence            = max(0.0, min(1.0, confidence))
        self.details               = details or {}
        self.error                 = error
        
        # Normalize probabilities to sum to 1
        total                      = self.synthetic_probability + self.authentic_probability + self.hybrid_probability
        
        if (total > 0):
            self.synthetic_probability /= total
            self.authentic_probability /= total
            self.hybrid_probability    /= total

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"metric_name"           : self.metric_name,
                "synthetic_probability" : round(self.synthetic_probability, 4),
                "authentic_probability" : round(self.authentic_probability, 4),
                "hybrid_probability"    : round(self.hybrid_probability, 4),
                "confidence"            : round(self.confidence, 4),
                "details"               : self.details,
                "error"                 : self.error,
                "success"               : self.error is None,
               }
    

    @property
    def is_synthetic(self) -> bool:
        """
        Check if classified as synthetic
        """
        return self.synthetic_probability > max(self.authentic_probability, self.hybrid_probability)
    
    
    @property
    def is_authentic(self) -> bool:
        """
        Check if classified as authentic
        """
        return self.authentic_probability > max(self.synthetic_probability, self.hybrid_probability)
    

    @property
    def is_hybrid(self) -> bool:
        """
        Check if classified as hybrid
        """
        return self.hybrid_probability > max(self.synthetic_probability, self.authentic_probability)
    

    @property
    def predicted_class(self) -> str:
        """
        Get predicted class
        """
        if self.is_synthetic:
            return "Synthetic"
        
        elif self.is_authentic:
            return "Authentic"
        
        else:
            return "Hybrid"


@dataclass
class EnsembleResult:
    """
    Result from ensemble classification
    """
    final_verdict          : str  # "Synthetically-Generated-Text", "Authentically-Written-Text", or "Hybrid-Text"
    synthetic_probability  : float
    authentic_probability  : float
    hybrid_probability     : float
    overall_confidence     : float
    domain                 : Domain
    metric_results         : Dict[str, MetricResult]
    metric_weights         : Dict[str, float]
    weighted_scores        : Dict[str, float]
    reasoning              : List[str]
    uncertainty_score      : float
    consensus_level        : float
    execution_mode         : str
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"final_verdict"         : self.final_verdict,
                "synthetic_probability" : round(self.synthetic_probability, 4),
                "authentic_probability" : round(self.authentic_probability, 4),
                "hybrid_probability"    : round(self.hybrid_probability, 4),
                "overall_confidence"    : round(self.overall_confidence, 4),
                "domain"                : self.domain.value,
                "uncertainty_score"     : round(self.uncertainty_score, 4),
                "consensus_level"       : round(self.consensus_level, 4),
                "metric_contributions"  : {name: {"weight"         : round(self.metric_weights.get(name, 0.0), 4),
                                                  "weighted_score" : round(self.weighted_scores.get(name, 0.0), 4),
                                                  "synthetic_prob" : round(result.synthetic_probability, 4),
                                                  "confidence"     : round(result.confidence, 4),
                                                 }
                                                 for name, result in self.metric_results.items()
                                         },
                "reasoning"             : self.reasoning,
                "execution_mode"        : self.execution_mode,
               }


@dataclass
class HighlightedSentenceResult:
    """
    A sentence with highlighting information
    """
    text                  : str
    synthetic_probability : float
    authentic_probability : float
    hybrid_probability    : float
    confidence            : float
    confidence_level      : ConfidenceLevel
    color_class           : str
    tooltip               : str
    index                 : int
    is_hybrid_content     : bool
    metric_breakdown      : Optional[Dict[str, float]] = None



@dataclass
class DetectionResult:
    """
    Complete detection result with all metadata
    """
    # Final results
    ensemble_result        : EnsembleResult
    
    # Input metadata
    processed_text         : ProcessedText
    domain_prediction      : DomainPrediction
    language_result        : Optional[LanguageDetectionResult]
    
    # Metric details
    metric_results         : Dict[str, MetricResult]
    
    # Performance metrics
    processing_time        : float
    metrics_execution_time : Dict[str, float] 
    
    # Warnings and errors
    warnings               : List[str]
    errors                 : List[str]
    
    # File information
    file_info              : Optional[Dict[str, Any]] = None
    
    # Execution mode
    execution_mode         : Optional[str]            = "parallel"
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        result = {"prediction"  : {"verdict"               : self.ensemble_result.final_verdict,
                                   "synthetic_probability" : round(self.ensemble_result.synthetic_probability, 4),
                                   "authentic_probability" : round(self.ensemble_result.authentic_probability, 4),
                                   "hybrid_probability"    : round(self.ensemble_result.hybrid_probability, 4),
                                   "confidence"            : round(self.ensemble_result.overall_confidence, 4),
                                  },
                  "analysis"    : {"domain"              : self.domain_prediction.primary_domain.value,
                                   "domain_confidence"   : round(self.domain_prediction.evidence_strength, 4),
                                   "language"            : self.language_result.primary_language.value if self.language_result else "unknown",
                                   "language_confidence" : round(self.language_result.evidence_strength, 4) if self.language_result else 0.0,
                                   "text_length"         : self.processed_text.word_count,
                                   "sentence_count"      : self.processed_text.sentence_count,
                                  },
                  "metrics"     : {name: result.to_dict() for name, result in self.metric_results.items()},
                  "ensemble"    : self.ensemble_result.to_dict(),
                  "performance" : {"total_time"   : round(self.processing_time, 3),
                                   "metrics_time" : {name: round(t, 3) for name, t in self.metrics_execution_time.items()},
                                  },
                  "warnings"    : self.warnings,
                  "errors"      : self.errors,
                 }
        
        # Include file_info if available
        if self.file_info:
            result["file_info"] = self.file_info
        
        return result


@dataclass
class DetailedReasoningResult:
    """
    Comprehensive reasoning for detection result with ensemble integration
    """
    summary                : str
    key_indicators         : List[str]
    metric_explanations    : Dict[str, str]
    supporting_evidence    : List[str]
    contradicting_evidence : List[str]
    confidence_explanation : str
    domain_analysis        : str
    ensemble_analysis      : str
    recommendations        : List[str]
    uncertainty_analysis   : str
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"summary"                : self.summary,
                "key_indicators"         : self.key_indicators,
                "metric_explanations"    : self.metric_explanations,
                "supporting_evidence"    : self.supporting_evidence,
                "contradicting_evidence" : self.contradicting_evidence,
                "confidence_explanation" : self.confidence_explanation,
                "domain_analysis"        : self.domain_analysis,
                "ensemble_analysis"      : self.ensemble_analysis,
                "recommendations"        : self.recommendations,
                "uncertainty_analysis"   : self.uncertainty_analysis,
               }


@dataclass
class DetailedMetricResult:
    """
    Metric data structure with sub-metrics
    """
    name                  : str
    synthetic_probability : float
    authentic_probability : float
    confidence            : float
    verdict               : str
    description           : str
    detailed_metrics      : Dict[str, float]
    weight                : float
