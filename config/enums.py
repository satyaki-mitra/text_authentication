# DEPENDENCIES
from enum import Enum


class ModelType(Enum):
    """
    Model types for categorization
    """
    TRANSFORMER             = "transformer"
    SENTENCE_TRANSFORMER    = "sentence_transformer"
    LANGUAGE_MODEL          = "language_model"
    MASKED_LANGUAGE_MODEL   = "masked_language_model"
    CLASSIFIER              = "classifier"
    EMBEDDING               = "embedding"
    RULE_BASED              = "rule_based"
    SEQUENCE_CLASSIFICATION = "sequence_classification"  
    CAUSAL_LM               = "causal_lm"        
    MASKED_LM               = "masked_lm" 


class Domain(Enum):
    """
    Text domains for adaptive thresholding
    """
    # Core domains
    GENERAL         = "general"
    ACADEMIC        = "academic"
    CREATIVE        = "creative"
    AI_ML           = "ai_ml"        # domain topic, not authorship        
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



class Language(Enum):
    """
    ISO 639-1 language codes for supported languages
    """
    ENGLISH    = "en"
    SPANISH    = "es"
    FRENCH     = "fr"
    GERMAN     = "de"
    ITALIAN    = "it"
    PORTUGUESE = "pt"
    RUSSIAN    = "ru"
    CHINESE    = "zh"
    JAPANESE   = "ja"
    KOREAN     = "ko"
    ARABIC     = "ar"
    HINDI      = "hi"
    DUTCH      = "nl"
    POLISH     = "pl"
    TURKISH    = "tr"
    SWEDISH    = "sv"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    THAI       = "th"
    GREEK      = "el"
    HEBREW     = "he"
    CZECH      = "cs"
    ROMANIAN   = "ro"
    DANISH     = "da"
    FINNISH    = "fi"
    NORWEGIAN  = "no"
    UNKNOWN    = "unknown"


class Script(Enum):
    """
    Writing scripts
    """
    LATIN      = "latin"
    CYRILLIC   = "cyrillic"
    ARABIC     = "arabic"
    CHINESE    = "chinese"
    JAPANESE   = "japanese"
    KOREAN     = "korean"
    DEVANAGARI = "devanagari"
    GREEK      = "greek"
    HEBREW     = "hebrew"
    THAI       = "thai"
    MIXED      = "mixed"
    UNKNOWN    = "unknown"



class ConfidenceLevel(Enum):
    """
    Confidence levels for authenticity estimation
    """
    VERY_LOW  = "very_low"
    LOW       = "low"
    MEDIUM    = "medium"
    HIGH      = "high"
    VERY_HIGH = "very_high"