# DEPENDENCIES
from enum import Enum
from typing import Dict
from typing import Tuple
from dataclasses import dataclass


class Domain(Enum):
    """
    Text domains for adaptive thresholding
    """
    # Core domains
    GENERAL         = "general"
    ACADEMIC        = "academic"
    CREATIVE        = "creative"
    AI_ML           = "ai_ml"              
    SOFTWARE_DEV    = "software_dev"       
    TECHNICAL_DOC   = "technical_doc"      
    ENGINEERING     = "engineering"   
    SCIENCE         = "science"     
    BUSINESS        = "business"
    LEGAL           = "legal"
    MEDICAL         = "medical"
    JOURNALISM      = "journalism"
    MARKETING       = "marketing"
    SOCIAL_MEDIA    = "social_media"
    BLOG_PERSONAL   = "blog_personal"   
    TUTORIAL        = "tutorial"      


class ConfidenceLevel(Enum):
    """
    Confidence levels for classification
    """
    VERY_LOW  = "very_low"
    LOW       = "low"
    MEDIUM    = "medium"
    HIGH      = "high"
    VERY_HIGH = "very_high"


@dataclass
class MetricThresholds:
    """
    Thresholds for a single metric
    """
    ai_threshold          : float       # Above this = likely AI
    human_threshold       : float       # Below this = likely human
    confidence_multiplier : float = 1.0
    weight                : float = 1.0


@dataclass
class DomainThresholds:
    """
    Thresholds for 6 metrics in a specific domain
    """
    domain             : Domain
    structural         : MetricThresholds
    perplexity         : MetricThresholds
    entropy            : MetricThresholds
    semantic_analysis  : MetricThresholds
    linguistic         : MetricThresholds
    detect_gpt         : MetricThresholds
    ensemble_threshold : float = 0.5


# ==================== DOMAIN-SPECIFIC THRESHOLDS ====================
# GENERAL (Default fallback)
DEFAULT_THRESHOLDS       = DomainThresholds(domain             = Domain.GENERAL,
                                            structural         = MetricThresholds(ai_threshold = 0.55, human_threshold = 0.45, weight = 0.20),
                                            perplexity         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.25),
                                            entropy            = MetricThresholds(ai_threshold = 0.48, human_threshold = 0.52, weight = 0.15),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.55, human_threshold = 0.45, weight = 0.18),
                                            linguistic         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.12),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.10),
                                            ensemble_threshold = 0.40,
                                           )

# ACADEMIC
ACADEMIC_THRESHOLDS      = DomainThresholds(domain             = Domain.ACADEMIC,
                                            structural         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.45, weight = 0.26),
                                            entropy            = MetricThresholds(ai_threshold = 0.45, human_threshold = 0.50, weight = 0.14),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.65, human_threshold = 0.35, weight = 0.08),
                                            ensemble_threshold = 0.42,
                                           )

# CREATIVE WRITING
CREATIVE_THRESHOLDS      = DomainThresholds(domain             = Domain.CREATIVE,
                                            structural         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.55, human_threshold = 0.50, weight = 0.22),
                                            entropy            = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.55, weight = 0.16),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.55, human_threshold = 0.45, weight = 0.16),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.08),
                                            ensemble_threshold = 0.38,
                                           )

# AI/ML/DATA SCIENCE
AI_ML_THRESHOLDS         = DomainThresholds(domain             = Domain.AI_ML,
                                            structural         = MetricThresholds(ai_threshold = 0.57, human_threshold = 0.43, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.51, human_threshold = 0.46, weight = 0.26),
                                            entropy            = MetricThresholds(ai_threshold = 0.47, human_threshold = 0.50, weight = 0.14),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.57, human_threshold = 0.43, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.61, human_threshold = 0.39, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.64, human_threshold = 0.36, weight = 0.08),
                                            ensemble_threshold = 0.41,
                                           )

# SOFTWARE DEVELOPMENT
SOFTWARE_DEV_THRESHOLDS  = DomainThresholds(domain             = Domain.SOFTWARE_DEV,
                                            structural         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.17),
                                            perplexity         = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.45, weight = 0.27),
                                            entropy            = MetricThresholds(ai_threshold = 0.46, human_threshold = 0.50, weight = 0.14),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.63, human_threshold = 0.37, weight = 0.08),
                                            ensemble_threshold = 0.41,
                                           )

# TECHNICAL DOCUMENTATION
TECHNICAL_DOC_THRESHOLDS = DomainThresholds(domain             = Domain.TECHNICAL_DOC,
                                            structural         = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.49, human_threshold = 0.44, weight = 0.27),
                                            entropy            = MetricThresholds(ai_threshold = 0.45, human_threshold = 0.49, weight = 0.13),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.65, human_threshold = 0.35, weight = 0.08),
                                            ensemble_threshold = 0.42,
                                           )

# ENGINEERING
ENGINEERING_THRESHOLDS   = DomainThresholds(domain             = Domain.ENGINEERING,
                                            structural         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.45, weight = 0.26),
                                            entropy            = MetricThresholds(ai_threshold = 0.46, human_threshold = 0.50, weight = 0.14),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.61, human_threshold = 0.39, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.64, human_threshold = 0.36, weight = 0.08),
                                            ensemble_threshold = 0.41,
                                           )

# SCIENCE (Physics, Chemistry, Biology)
SCIENCE_THRESHOLDS       = DomainThresholds(domain             = Domain.SCIENCE,
                                            structural         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.51, human_threshold = 0.46, weight = 0.26),
                                            entropy            = MetricThresholds(ai_threshold = 0.46, human_threshold = 0.50, weight = 0.14),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.14),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.64, human_threshold = 0.36, weight = 0.08),
                                            ensemble_threshold = 0.42,
                                           )

# BUSINESS
BUSINESS_THRESHOLDS      = DomainThresholds(domain             = Domain.BUSINESS,
                                            structural         = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.24),
                                            entropy            = MetricThresholds(ai_threshold = 0.48, human_threshold = 0.52, weight = 0.15),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.19),
                                            linguistic         = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.15),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.09),
                                            ensemble_threshold = 0.40,
                                           )

# LEGAL
LEGAL_THRESHOLDS         = DomainThresholds(domain             = Domain.LEGAL,
                                            structural         = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.17),
                                            perplexity         = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.44, weight = 0.27),
                                            entropy            = MetricThresholds(ai_threshold = 0.44, human_threshold = 0.48, weight = 0.13),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.63, human_threshold = 0.37, weight = 0.15),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.66, human_threshold = 0.34, weight = 0.08),
                                            ensemble_threshold = 0.43,
                                           )

# MEDICAL
MEDICAL_THRESHOLDS       = DomainThresholds(domain             = Domain.MEDICAL,
                                            structural         = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.17),
                                            perplexity         = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.45, weight = 0.27),
                                            entropy            = MetricThresholds(ai_threshold = 0.45, human_threshold = 0.49, weight = 0.13),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.15),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.65, human_threshold = 0.35, weight = 0.08),
                                            ensemble_threshold = 0.43,
                                           )

# JOURNALISM
JOURNALISM_THRESHOLDS    = DomainThresholds(domain             = Domain.JOURNALISM,
                                            structural         = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.24),
                                            entropy            = MetricThresholds(ai_threshold = 0.48, human_threshold = 0.52, weight = 0.15),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.20),
                                            linguistic         = MetricThresholds(ai_threshold = 0.58, human_threshold = 0.42, weight = 0.15),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.08),
                                            ensemble_threshold = 0.40,
                                           )

# MARKETING
MARKETING_THRESHOLDS     = DomainThresholds(domain             = Domain.MARKETING,
                                            structural         = MetricThresholds(ai_threshold = 0.54, human_threshold = 0.46, weight = 0.19),
                                            perplexity         = MetricThresholds(ai_threshold = 0.53, human_threshold = 0.49, weight = 0.23),
                                            entropy            = MetricThresholds(ai_threshold = 0.49, human_threshold = 0.53, weight = 0.15),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.54, human_threshold = 0.46, weight = 0.19),
                                            linguistic         = MetricThresholds(ai_threshold = 0.57, human_threshold = 0.43, weight = 0.16),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.61, human_threshold = 0.39, weight = 0.08),
                                            ensemble_threshold = 0.39,
                                           )

# SOCIAL MEDIA
SOCIAL_MEDIA_THRESHOLDS  = DomainThresholds(domain             = Domain.SOCIAL_MEDIA,
                                            structural         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.54, human_threshold = 0.50, weight = 0.20),
                                            entropy            = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.54, weight = 0.17),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.18),
                                            linguistic         = MetricThresholds(ai_threshold = 0.55, human_threshold = 0.45, weight = 0.18),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.60, human_threshold = 0.40, weight = 0.09),
                                            ensemble_threshold = 0.36,
                                           )

# PERSONAL BLOG
BLOG_PERSONAL_THRESHOLDS = DomainThresholds(domain             = Domain.BLOG_PERSONAL,
                                            structural         = MetricThresholds(ai_threshold = 0.53, human_threshold = 0.47, weight = 0.19),
                                            perplexity         = MetricThresholds(ai_threshold = 0.54, human_threshold = 0.50, weight = 0.22),
                                            entropy            = MetricThresholds(ai_threshold = 0.50, human_threshold = 0.54, weight = 0.16),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.53, human_threshold = 0.47, weight = 0.19),
                                            linguistic         = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.16),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.08),
                                            ensemble_threshold = 0.38,
                                           )

# TUTORIAL/HOW-TO
TUTORIAL_THRESHOLDS      = DomainThresholds(domain             = Domain.TUTORIAL,
                                            structural         = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.18),
                                            perplexity         = MetricThresholds(ai_threshold = 0.52, human_threshold = 0.48, weight = 0.25),
                                            entropy            = MetricThresholds(ai_threshold = 0.48, human_threshold = 0.52, weight = 0.15),
                                            semantic_analysis  = MetricThresholds(ai_threshold = 0.56, human_threshold = 0.44, weight = 0.19),
                                            linguistic         = MetricThresholds(ai_threshold = 0.59, human_threshold = 0.41, weight = 0.15),
                                            detect_gpt         = MetricThresholds(ai_threshold = 0.62, human_threshold = 0.38, weight = 0.08),
                                            ensemble_threshold = 0.40,
                                           )


# THRESHOLD REGISTRY
THRESHOLD_REGISTRY: Dict[Domain, DomainThresholds]            = {Domain.GENERAL       : DEFAULT_THRESHOLDS,
                                                                 Domain.ACADEMIC      : ACADEMIC_THRESHOLDS,
                                                                 Domain.CREATIVE      : CREATIVE_THRESHOLDS,
                                                                 Domain.AI_ML         : AI_ML_THRESHOLDS,
                                                                 Domain.SOFTWARE_DEV  : SOFTWARE_DEV_THRESHOLDS,
                                                                 Domain.TECHNICAL_DOC : TECHNICAL_DOC_THRESHOLDS,
                                                                 Domain.ENGINEERING   : ENGINEERING_THRESHOLDS,
                                                                 Domain.SCIENCE       : SCIENCE_THRESHOLDS,
                                                                 Domain.BUSINESS      : BUSINESS_THRESHOLDS,
                                                                 Domain.LEGAL         : LEGAL_THRESHOLDS,
                                                                 Domain.MEDICAL       : MEDICAL_THRESHOLDS,
                                                                 Domain.JOURNALISM    : JOURNALISM_THRESHOLDS,
                                                                 Domain.MARKETING     : MARKETING_THRESHOLDS,
                                                                 Domain.SOCIAL_MEDIA  : SOCIAL_MEDIA_THRESHOLDS,
                                                                 Domain.BLOG_PERSONAL : BLOG_PERSONAL_THRESHOLDS,
                                                                 Domain.TUTORIAL      : TUTORIAL_THRESHOLDS,
                                                                }


# CONFIDENCE LEVEL RANGES
CONFIDENCE_RANGES: Dict[ConfidenceLevel, Tuple[float, float]] = {ConfidenceLevel.VERY_LOW  : (0.0, 0.3),
                                                                 ConfidenceLevel.LOW       : (0.3, 0.5),
                                                                 ConfidenceLevel.MEDIUM    : (0.5, 0.7),
                                                                 ConfidenceLevel.HIGH      : (0.7, 0.85),
                                                                 ConfidenceLevel.VERY_HIGH : (0.85, 1.0),
                                                                }


# HELPER FUNCTIONS 
def get_threshold_for_domain(domain: Domain) -> DomainThresholds:
    """
    Get thresholds for a specific domain
    """
    return THRESHOLD_REGISTRY.get(domain, DEFAULT_THRESHOLDS)


def get_confidence_level(score: float) -> ConfidenceLevel:
    """
    Determine confidence level based on score
    """
    for level, (min_val, max_val) in CONFIDENCE_RANGES.items():
        if (min_val <= score < max_val):
            return level

    return ConfidenceLevel.VERY_HIGH


def adjust_threshold_by_confidence(threshold: float, confidence: float, conservative: bool = True) -> float:
    """
    Adjust threshold based on confidence level
    """
    if conservative:
        adjustment         = (1 - confidence) * 0.1
        adjusted_threshold = threshold + adjustment
        
        return adjusted_threshold

    else:
        adjustment         = confidence * 0.05
        adjusted_threshold = threshold - adjustment
        
        return adjusted_threshold


def interpolate_thresholds(domain1: Domain, domain2: Domain, weight1: float = 0.5) -> DomainThresholds:
    """
    Interpolate between two domain thresholds
    """
    thresh1 = get_threshold_for_domain(domain = domain1)
    thresh2 = get_threshold_for_domain(domain = domain2)
    weight2 = 1 - weight1
    
    def interpolate_metric(m1: MetricThresholds, m2: MetricThresholds) -> MetricThresholds:
        return MetricThresholds(ai_threshold    = m1.ai_threshold * weight1 + m2.ai_threshold * weight2,
                                human_threshold = m1.human_threshold * weight1 + m2.human_threshold * weight2,
                                weight          = m1.weight * weight1 + m2.weight * weight2,
                               )
    
    return DomainThresholds(domain              = domain1,
                            structural          = interpolate_metric(thresh1.structural, thresh2.structural),
                            perplexity          = interpolate_metric(thresh1.perplexity, thresh2.perplexity),
                            entropy             = interpolate_metric(thresh1.entropy, thresh2.entropy),
                            semantic_analysis   = interpolate_metric(thresh1.semantic_analysis, thresh2.semantic_analysis),
                            linguistic          = interpolate_metric(thresh1.linguistic, thresh2.linguistic),
                            detect_gpt          = interpolate_metric(thresh1.detect_gpt, thresh2.detect_gpt),
                            ensemble_threshold  = thresh1.ensemble_threshold * weight1 + thresh2.ensemble_threshold * weight2,
                           )


def get_active_metric_weights(domain: Domain, enabled_metrics: Dict[str, bool]) -> Dict[str, float]:
    """
    Get weights for enabled metrics, normalized to sum to 1.0
    """
    thresholds     = get_threshold_for_domain(domain = domain)
    
    metric_mapping = {"structural"        : thresholds.structural,
                      "perplexity"        : thresholds.perplexity,
                      "entropy"           : thresholds.entropy,
                      "semantic_analysis" : thresholds.semantic_analysis,
                      "linguistic"        : thresholds.linguistic,
                      "detect_gpt"        : thresholds.detect_gpt,
                     }
    
    active_weights = dict()

    for metric_name, threshold_obj in metric_mapping.items():
        if enabled_metrics.get(metric_name, False):
            active_weights[metric_name] = threshold_obj.weight
    
    # Normalize
    total_weight = sum(active_weights.values())

    if (total_weight > 0):
        active_weights = {name: weight / total_weight for name, weight in active_weights.items()}
    
    return active_weights



# Export
__all__ = ["Domain",
           "ConfidenceLevel",
           "MetricThresholds",
           "DomainThresholds",
           "CONFIDENCE_RANGES",
           "DEFAULT_THRESHOLDS",
           "THRESHOLD_REGISTRY",
           "get_confidence_level",
           "interpolate_thresholds",  
           "get_threshold_for_domain",           
           "get_active_metric_weights",
           "adjust_threshold_by_confidence",
          ]