# DEPENDENCIES
from typing import Dict
from typing import List
from typing import Tuple
from dataclasses import field
from config.enums import Script
from dataclasses import dataclass


@dataclass(frozen = True)
class DocumentExtractionParams:
    """
    Hyperparameters for Document Extraction
    """
    # Supported file extensions
    SUPPORTED_EXTENSIONS : frozenset = frozenset({'.txt', '.text', '.md', '.markdown', '.log', '.csv', '.pdf', '.docx', '.doc',  '.rtf', '.html', '.htm'})
        
    # Text file extensions
    TEXT_EXTENSIONS      : frozenset = frozenset({'.txt', '.text', '.md', '.markdown', '.log', '.csv'})
        
    # Maximum file size (50 MB default)
    MAX_FILE_SIZE        : int       = 50 * 1024 * 1024



@dataclass(frozen = True)
class LanguageDetectionParams:
    """
    Hyperparameters for Language Detection
    """
    # Text length constraints
    MINIMUM_TEXT_LENGTH             : int                              = 20
    
    # Chunking parameters
    MAX_CHUNK_LENGTH                : int                              = 500
    MIN_CHUNK_LENGTH                : int                              = 50
    FIXED_CHUNK_SIZE                : int                              = 1000
    
    # Model parameters
    MODEL_MAX_LENGTH                : int                              = 512
    TOP_K_PREDICTIONS               : int                              = 3
    
    # Confidence thresholds
    LOW_CONFIDENCE_THRESHOLD        : float                            = 0.6
    MULTILINGUAL_THRESHOLD          : float                            = 0.2
    SCRIPT_DOMINANCE_THRESHOLD      : float                            = 0.7
    LANGUAGE_MATCH_THRESHOLD        : float                            = 0.7
    
    # Quality assessment
    WORD_BOUNDARY_RATIO             : float                            = 0.7
    MIXED_DOMAIN_CONFIDENCE_PENALTY : float                            = 0.8
    
    # Language name mappings
    LANGUAGE_NAMES                  : Dict[str, str]                   = field(default_factory = lambda : {"en": "English",
                                                                                                           "es": "Spanish",
                                                                                                           "fr": "French",
                                                                                                           "de": "German",
                                                                                                           "it": "Italian",
                                                                                                           "pt": "Portuguese",
                                                                                                           "ru": "Russian",
                                                                                                           "zh": "Chinese",
                                                                                                           "ja": "Japanese",
                                                                                                           "ko": "Korean",
                                                                                                           "ar": "Arabic",
                                                                                                           "hi": "Hindi",
                                                                                                          }
                                                                              )
    
    # Unicode script ranges
    SCRIPT_RANGES                   : Dict[str, List[Tuple[int, int]]] = field(default_factory = lambda: {"latin"      : [(0x0041, 0x007A), (0x00C0, 0x024F)],
                                                                                                          "cyrillic"   : [(0x0400, 0x04FF)],
                                                                                                          "arabic"     : [(0x0600, 0x06FF), (0x0750, 0x077F)],
                                                                                                          "chinese"    : [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
                                                                                                          "japanese"   : [(0x3040, 0x309F), (0x30A0, 0x30FF)],
                                                                                                          "korean"     : [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
                                                                                                          "devanagari" : [(0x0900, 0x097F)],
                                                                                                          "greek"      : [(0x0370, 0x03FF)],
                                                                                                          "hebrew"     : [(0x0590, 0x05FF)],
                                                                                                          "thai"       : [(0x0E00, 0x0E7F)],
                                                                                                         }
                                                                              )




@dataclass(frozen = True)
class TextProcessingParams:
    """
    Hyperparameters for Text Processing
    """
    # Text length constraints
    MINIMUM_TEXT_LENGTH     : int    = 20
    MAXIMUM_TEXT_LENGTH     : int    = 1000000  # 1M characters
    
    # Text cleaning options
    PRESERVE_FORMATTING     : bool   = False
    REMOVE_URLS             : bool   = True
    REMOVE_EMAILS           : bool   = True
    NORMALIZE_UNICODE       : bool   = True
    FIX_ENCODING            : bool   = True
    
    # Validation thresholds
    MINIMUM_WORD_COUNT      : int    = 10
    
    # Common abbreviations for sentence splitting
    COMMON_ABBREVIATIONS    : list   = field(default_factory  = lambda: ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Gen.", "Sen.", "Rep.", "St.", "Ave.", "Blvd.", "Rd.", "Pkwy.", "Co.", "Ltd.", "Inc.", "Corp.", 
                                                                         "vs.", "etc.", "e.g.", "i.e.", "c.", "ca.", "cf.", "al.", "et al.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", 
                                                                         "Nov.", "Dec.", "Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun.", "kg.", "g.", "mg.", "km.", "m.", "cm.", "mm.", "hr.", "min.", "sec.", 
                                                                         "vol.", "no.", "p.", "pp.", "ch.", "fig.", "ed.", "trans.", "approx.", "est.", "max.", "min.", "avg.", "std.", "temp.", "pres.", "vol.", "ibid.",
                                                                         "op.", "cit.", "loc.", "cf.", "viz.", "sc.", "seq."
                                                                        ]
                                            )


@dataclass(frozen = True)
class DomainClassificationParams:
    """
    Hyperparameters for Domain Classification
    """
    # Classification parameters
    TOP_K_DOMAINS                   : int                  = 2
    MIN_CONFIDENCE_THRESHOLD        : float                = 0.3
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD       : float                = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD     : float                = 0.6
    LOW_CONFIDENCE_THRESHOLD        : float                = 0.5
    SECONDARY_DOMAIN_MIN_SCORE      : float                = 0.1
    
    # Mixed domain detection
    MIXED_DOMAIN_PRIMARY_MAX        : float                = 0.7
    MIXED_DOMAIN_SECONDARY_MIN      : float                = 0.3
    MIXED_DOMAIN_RATIO_THRESHOLD    : float                = 0.6
    MIXED_DOMAIN_CONFIDENCE_PENALTY : float                = 0.8
    
    # Text preprocessing
    MAX_WORDS_FOR_CLASSIFICATION    : int                  = 400
    
    # Domain labels for zero-shot classification
    DOMAIN_LABELS                   : Dict[str, List[str]] = field(default_factory = lambda : {"academic"      : ["academic paper", "research article", "scientific paper", "scholarly writing", "thesis", "dissertation", "academic research"],
                                                                                               "creative"      : ["creative writing", "fiction", "story", "narrative", "poetry", "literary work", "imaginative writing"],
                                                                                               "ai_ml"         : ["artificial intelligence", "machine learning", "neural networks", "data science", "AI research", "deep learning"],
                                                                                               "software_dev"  : ["software development", "programming", "coding", "software engineering", "web development", "application development"],
                                                                                               "technical_doc" : ["technical documentation", "user manual", "API documentation", "technical guide", "system documentation"],
                                                                                               "engineering"   : ["engineering document", "technical design", "engineering analysis", "mechanical engineering", "electrical engineering"],
                                                                                               "science"       : ["scientific research", "physics", "chemistry", "biology", "scientific study", "experimental results"],
                                                                                               "business"      : ["business document", "corporate communication", "business report", "professional writing", "executive summary"],
                                                                                               "journalism"    : ["news article", "journalism", "press release", "news report", "media content", "reporting"],
                                                                                               "social_media"  : ["social media post", "casual writing", "online content", "informal text", "social media content"],
                                                                                               "blog_personal" : ["personal blog", "personal writing", "lifestyle blog", "personal experience", "opinion piece", "diary entry"],
                                                                                               "legal"         : ["legal document", "contract", "legal writing", "law", "legal agreement", "legal analysis"],
                                                                                               "medical"       : ["medical document", "healthcare", "clinical", "medical report", "health information", "medical research"],
                                                                                               "marketing"     : ["marketing content", "advertising", "brand content", "promotional writing", "sales copy", "marketing material"],
                                                                                               "tutorial"      : ["tutorial", "how-to guide", "instructional content", "step-by-step guide", "educational guide", "learning material"],
                                                                                               "general"       : ["general content", "everyday writing", "common text", "standard writing", "normal text", "general information"],
                                                                                              }
                                                                  )


@dataclass(frozen = True)
class BaseMetricParams:
    """
    Hyperparameters for BaseMetric class
    """
    DEFAULT_AUTHENTIC_PROBABILITY : float = 0.5
    DEFAULT_SYNTHETIC_PROBABILITY : float = 0.5
    DEFAULT_HYBRID_PROBABILITY    : float = 0.0
    DEFAULT_CONFIDENCE            : float = 0.0


@dataclass(frozen = True)
class StructuralMetricParams:
    """
    Hyperparameters for Structural Metric
    """
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB       : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB       : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT      : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START  : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START  : float = 0.7
    UNCERTAIN_RANGE_WIDTH            : float = 0.4
    NEUTRAL_PROBABILITY              : float = 0.5  # For fallback
    MIN_PROBABILITY                  : float = 0.0
    MAX_PROBABILITY                  : float = 1.0
    
    # Feature extraction - sentence splitting
    SENTENCE_SPLIT_PATTERN           : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    WORD_TOKENIZE_PATTERN            : str   = r'\b\w+\b'
    PUNCTUATION_PATTERN              : str   = r'[^\w\s]'
    
    # Burstiness calculation
    BURSTINESS_NORMALIZATION_FACTOR  : float = 2.0
    
    # Readability calculation
    FLESCH_CONSTANT_1                : float = 206.835
    FLESCH_CONSTANT_2                : float = 1.015
    FLESCH_CONSTANT_3                : float = 84.6
    NEUTRAL_READABILITY_SCORE        : float = 50.0
    MIN_READABILITY_SCORE            : float = 0.0
    MAX_READABILITY_SCORE            : float = 100.0
    
    # Repetition detection
    REPETITION_WINDOW_SIZE           : int   = 10
    MIN_WORDS_FOR_REPETITION         : int   = 10
    
    # N-gram analysis
    BIGRAM_N                         : int   = 2
    TRIGRAM_N                        : int   = 3
    
    # Synthetic probability calculation thresholds
    BURSTINESS_LOW_THRESHOLD         : float = 0.3
    BURSTINESS_MEDIUM_THRESHOLD      : float = 0.5
    LENGTH_UNIFORMITY_HIGH_THRESHOLD : float = 0.7
    LENGTH_UNIFORMITY_MEDIUM_THRESH  : float = 0.5
    BIGRAM_DIVERSITY_LOW_THRESHOLD   : float = 0.7
    READABILITY_SYNTHETIC_MIN        : float = 60.0
    READABILITY_SYNTHETIC_MAX        : float = 75.0
    REPETITION_LOW_THRESHOLD         : float = 0.1
    REPETITION_MEDIUM_THRESHOLD      : float = 0.2
    
    # Synthetic probability weights
    STRONG_SYNTHETIC_WEIGHT          : float = 0.7
    MODERATE_SYNTHETIC_WEIGHT        : float = 0.5
    WEAK_SYNTHETIC_WEIGHT            : float = 0.3
    VERY_WEAK_SYNTHETIC_WEIGHT       : float = 0.4
    NEUTRAL_WEIGHT                   : float = 0.5
    
    # Confidence calculation
    CONFIDENCE_STD_NORMALIZER        : float = 0.5
    MIN_CONFIDENCE                   : float = 0.1
    MAX_CONFIDENCE                   : float = 0.9
    NEUTRAL_CONFIDENCE               : float = 0.5  # For fallback
    
    # Hybrid probability calculation
    BURSTINESS_HIGH_THRESHOLD        : float = 0.6
    SENTENCE_LENGTH_VARIANCE_RATIO   : float = 0.8
    TYPE_TOKEN_RATIO_EXTREME_LOW     : float = 0.3
    TYPE_TOKEN_RATIO_EXTREME_HIGH    : float = 0.9
    READABILITY_EXTREME_LOW          : float = 20.0
    READABILITY_EXTREME_HIGH         : float = 90.0
    MODERATE_HYBRID_WEIGHT           : float = 0.4
    WEAK_HYBRID_WEIGHT               : float = 0.3
    MAX_HYBRID_PROBABILITY           : float = 0.3
    
    # Feature validation
    MIN_SENTENCE_LENGTH_FOR_STD      : int   = 2
    MIN_WORD_LENGTH_FOR_STD          : int   = 2
    MIN_VALUES_FOR_BURSTINESS        : int   = 2
    MIN_WORDS_FOR_NGRAM              : int   = 2  # For n-gram where n=2
    
    # Math and normalization
    ZERO_TOLERANCE                   : float = 1e-10
    ZERO_VALUE                       : float = 0.0
    ONE_VALUE                        : float = 1.0


@dataclass(frozen = True)
class SemanticAnalysisParams:
    """
    Hyperparameters for Semantic Analysis Metric
    """
    # Text validation
    MIN_TEXT_LENGTH_FOR_ANALYSIS        : int   = 50
    MIN_SENTENCES_FOR_ANALYSIS          : int   = 3
    MIN_SENTENCE_LENGTH                 : int   = 10
    MIN_VALID_SENTENCE_LENGTH           : int   = 5
    
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB          : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB          : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT         : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START     : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START     : float = 0.7
    UNCERTAIN_RANGE_WIDTH               : float = 0.4
    NEUTRAL_PROBABILITY                 : float = 0.5
    MIN_PROBABILITY                     : float = 0.0
    MAX_PROBABILITY                     : float = 1.0
    
    # Sentence splitting
    SENTENCE_SPLIT_PATTERN              : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    WORD_EXTRACTION_PATTERN             : str   = r'\b[a-zA-Z]{4,}\b'
    
    # Coherence calculation
    HIGH_COHERENCE_SIMILARITY           : float = 0.8
    SIMILARITY_VARIANCE_FACTOR          : float = 5.0
    
    # Repetition detection
    REPETITION_SIMILARITY_THRESHOLD     : float = 0.8
    REPETITION_SCORE_SCALING            : float = 3.0
    MIN_SENTENCES_FOR_REPETITION        : int   = 5
    
    # Topic drift calculation
    START_SECTION_SIZE                  : int   = 3
    END_SECTION_SIZE                    : int   = 3
    SECTION_SIZE_RATIO                  : int   = 3  # denominator for section size calculation
    
    # Chunk analysis 
    CHUNK_SIZE_WORDS                    : int   = 200
    CHUNK_OVERLAP_RATIO                 : float = 0.5  # 50% overlap
    MIN_CHUNK_LENGTH                    : int   = 50
    MIN_SENTENCES_PER_CHUNK             : int   = 2
    
    # Keyword analysis
    MIN_WORDS_FOR_KEYWORD_ANALYSIS      : int   = 10
    TOP_KEYWORDS_COUNT                  : int   = 10
    MIN_KEYWORD_FREQUENCY               : int   = 2
    
    # Synthetic probability thresholds
    COHERENCE_HIGH_THRESHOLD            : float = 0.7
    COHERENCE_MEDIUM_THRESHOLD          : float = 0.5
    CONSISTENCY_HIGH_THRESHOLD          : float = 0.8
    CONSISTENCY_MEDIUM_THRESHOLD        : float = 0.6
    REPETITION_HIGH_THRESHOLD           : float = 0.3
    REPETITION_MEDIUM_THRESHOLD         : float = 0.1
    TOPIC_DRIFT_LOW_THRESHOLD           : float = 0.2
    TOPIC_DRIFT_MEDIUM_THRESHOLD        : float = 0.4
    COHERENCE_VARIANCE_LOW_THRESHOLD    : float = 0.05
    COHERENCE_VARIANCE_MEDIUM_THRESHOLD : float = 0.1
    
    # Synthetic probability weights
    STRONG_SYNTHETIC_WEIGHT             : float = 0.9
    MODERATE_SYNTHETIC_WEIGHT           : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT             : float = 0.6
    WEAK_SYNTHETIC_WEIGHT               : float = 0.5
    VERY_WEAK_SYNTHETIC_WEIGHT          : float = 0.4
    VERY_LOW_SYNTHETIC_WEIGHT           : float = 0.3
    LOW_SYNTHETIC_WEIGHT                : float = 0.2
    
    # Confidence calculation
    CONFIDENCE_STD_NORMALIZER           : float = 0.5
    MIN_CONFIDENCE                      : float = 0.1
    MAX_CONFIDENCE                      : float = 0.9
    NEUTRAL_CONFIDENCE                  : float = 0.5
    LOW_FEATURE_CONFIDENCE              : float = 0.3
    
    # Hybrid probability calculation
    COHERENCE_MIXED_MIN                 : float = 0.4
    COHERENCE_MIXED_MAX                 : float = 0.6
    COHERENCE_VARIANCE_HIGH_THRESHOLD   : float = 0.15
    COHERENCE_VARIANCE_MEDIUM_THRESHOLD : float = 0.1
    REPETITION_MIXED_MIN                : float = 0.15
    REPETITION_MIXED_MAX                : float = 0.35
    MODERATE_HYBRID_WEIGHT              : float = 0.4
    WEAK_HYBRID_WEIGHT                  : float = 0.3
    VERY_WEAK_HYBRID_WEIGHT             : float = 0.2
    MAX_HYBRID_PROBABILITY              : float = 0.3
    
    # Default feature values
    DEFAULT_COHERENCE                   : float = 0.5
    DEFAULT_CONSISTENCY                 : float = 0.5
    DEFAULT_REPETITION                  : float = 0.0
    DEFAULT_TOPIC_DRIFT                 : float = 0.5
    DEFAULT_CONTEXTUAL_CONSISTENCY      : float = 0.5
    DEFAULT_CHUNK_COHERENCE             : float = 0.5
    DEFAULT_COHERENCE_VARIANCE          : float = 0.1
    
    # Error handling
    MIN_REQUIRED_FEATURES               : int   = 3
    ZERO_TOLERANCE                      : float = 1e-10


@dataclass(frozen = True)
class LinguisticMetricParams:
    """
    Hyperparameters for Linguistic Metric
    """
    # Text validation
    MIN_TEXT_LENGTH_FOR_ANALYSIS             : int   = 50
    
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB               : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB               : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT              : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START          : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START          : float = 0.7
    UNCERTAIN_RANGE_WIDTH                    : float = 0.4
    NEUTRAL_PROBABILITY                      : float = 0.5
    MIN_PROBABILITY                          : float = 0.0
    MAX_PROBABILITY                          : float = 1.0
    
    # POS analysis
    MIN_TAGS_FOR_ENTROPY                     : int   = 1
    
    # Syntactic complexity
    COMPLEXITY_WEIGHT_AVG                    : float = 0.5
    COMPLEXITY_WEIGHT_MAX                    : float = 0.5
    
    # Sentence complexity
    WORDS_PER_COMPLEXITY_UNIT                : float = 10.0
    CLAUSE_COMPLEXITY_FACTOR                 : float = 0.5
    
    # Grammatical patterns
    TRANSITION_WORDS_SET                     : tuple = ('however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally', 'nevertheless', 'nonetheless', 'thus', 'hence')
    IDEAL_PASSIVE_RATIO                      : float = 0.3
    IDEAL_TRANSITION_RATIO                   : float = 0.2
    PASSIVE_DEPENDENCY                       : str   = 'nsubjpass'
    CLAUSE_MARKERS                           : tuple = ('cc', 'mark')
    
    # Writing style analysis
    IDEAL_LENGTH_VARIATION                   : float = 0.5
    IDEAL_PUNCTUATION_RATIO                  : float = 0.1
    
    # SYNTHETIC pattern detection
    TRANSITION_OVERUSE_THRESHOLD             : float = 0.05
    POS_SEQUENCE_FREQ_THRESHOLD              : float = 0.1
    STRUCTURE_DIVERSITY_THRESHOLD            : float = 0.5
    UNUSUAL_CONSTRUCTION_THRESHOLD           : float = 0.02
    REPETITIVE_PHRASING_THRESHOLD            : float = 0.3
    UNUSUAL_DEPENDENCIES                     : tuple = ('attr', 'oprd')
    
    # Chunk analysis
    CHUNK_SIZE_WORDS                         : int   = 200
    CHUNK_OVERLAP_RATIO                      : float = 0.5
    MIN_CHUNK_LENGTH                         : int   = 50
    MIN_SENTENCES_FOR_STRUCTURE              : int   = 3
    MIN_SENTENCES_FOR_ANALYSIS               : int   = 1
    
    # Synthetic probability thresholds
    POS_DIVERSITY_LOW_THRESHOLD              : float = 0.3
    POS_DIVERSITY_MEDIUM_THRESHOLD           : float = 0.5
    SYNTACTIC_COMPLEXITY_LOW_THRESHOLD       : float = 2.0
    SYNTACTIC_COMPLEXITY_MEDIUM_THRESHOLD    : float = 3.0
    GRAMMATICAL_CONSISTENCY_HIGH_THRESHOLD   : float = 0.8
    GRAMMATICAL_CONSISTENCY_MEDIUM_THRESHOLD : float = 0.6
    TRANSITION_USAGE_HIGH_THRESHOLD          : float = 0.3
    TRANSITION_USAGE_MEDIUM_THRESHOLD        : float = 0.15
    SYNTHETIC_PATTERN_HIGH_THRESHOLD         : float = 0.6
    SYNTHETIC_PATTERN_MEDIUM_THRESHOLD       : float = 0.3
    COMPLEXITY_VARIANCE_LOW_THRESHOLD        : float = 0.1
    COMPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 0.3
    
    # Synthetic probability weights
    STRONG_SYNTHETIC_WEIGHT                  : float = 0.9
    MODERATE_SYNTHETIC_WEIGHT                : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT                  : float = 0.7
    WEAK_SYNTHETIC_WEIGHT                    : float = 0.6
    VERY_WEAK_SYNTHETIC_WEIGHT               : float = 0.5
    LOW_SYNTHETIC_WEIGHT                     : float = 0.4
    VERY_LOW_SYNTHETIC_WEIGHT                : float = 0.3
    MINIMAL_SYNTHETIC_WEIGHT                 : float = 0.2
    
    # Confidence calculation
    CONFIDENCE_STD_NORMALIZER                : float = 0.5
    MIN_CONFIDENCE                           : float = 0.1
    MAX_CONFIDENCE                           : float = 0.9
    NEUTRAL_CONFIDENCE                       : float = 0.5
    LOW_FEATURE_CONFIDENCE                   : float = 0.3
    MIN_REQUIRED_FEATURES                    : int   = 4
    
    # Hybrid probability calculation
    POS_DIVERSITY_MIXED_MIN                  : float = 0.35
    POS_DIVERSITY_MIXED_MAX                  : float = 0.55
    POS_ENTROPY_LOW_THRESHOLD                : float = 0.35
    POS_ENTROPY_HIGH_THRESHOLD               : float = 0.65
    COMPLEXITY_VARIANCE_HIGH_THRESHOLD       : float = 0.5
    COMPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 0.3
    SYNTHETIC_PATTERN_MIXED_MIN              : float = 0.2
    SYNTHETIC_PATTERN_MIXED_MAX              : float = 0.6
    MODERATE_HYBRID_WEIGHT                   : float = 0.4
    WEAK_HYBRID_WEIGHT                       : float = 0.3
    MINIMAL_HYBRID_WEIGHT                    : float = 0.2
    MAX_HYBRID_PROBABILITY                   : float = 0.3
    
    # Default feature values
    DEFAULT_POS_DIVERSITY                    : float = 0.5
    DEFAULT_POS_ENTROPY                      : float = 2.5
    DEFAULT_SYNTACTIC_COMPLEXITY             : float = 2.5
    DEFAULT_SENTENCE_COMPLEXITY              : float = 2.0
    DEFAULT_GRAMMATICAL_CONSISTENCY          : float = 0.5
    DEFAULT_TRANSITION_USAGE                 : float = 0.1
    DEFAULT_PASSIVE_RATIO                    : float = 0.2
    DEFAULT_WRITING_STYLE_SCORE              : float = 0.5
    DEFAULT_SYNTHETIC_PATTERN_SCORE          : float = 0.3
    DEFAULT_CHUNK_COMPLEXITY                 : float = 2.5
    DEFAULT_COMPLEXITY_VARIANCE              : float = 0.2
    
    # Math and normalization
    LOG_BASE                                 : int   = 2
    ZERO_TOLERANCE                           : float = 1e-10


@dataclass(frozen = True)
class PerplexityMetricParams:
    """
    Hyperparameters for Perplexity Metric
    """
    # Text validation
    MIN_TEXT_LENGTH_FOR_ANALYSIS             : int   = 50
    
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB               : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB               : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT              : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START          : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START          : float = 0.7
    UNCERTAIN_RANGE_WIDTH                    : float = 0.4
    NEUTRAL_PROBABILITY                      : float = 0.5
    MIN_PROBABILITY                          : float = 0.0
    MAX_PROBABILITY                          : float = 1.0
    
    # Model parameters
    MAX_TOKEN_LENGTH                         : int   = 1024
    MIN_TOKENS_FOR_PERPLEXITY                : int   = 5
    MIN_SENTENCE_LENGTH                      : int   = 20
    MIN_CHUNK_LENGTH                         : int   = 50
    
    # Chunk analysis
    CHUNK_SIZE_WORDS                         : int   = 200
    CHUNK_OVERLAP_RATIO                      : float = 0.5
    
    # Perplexity normalization
    PERPLEXITY_SIGMOID_CENTER                : float = 30.0
    PERPLEXITY_SIGMOID_SCALE                 : float = 10.0
    
    # Cross-entropy normalization
    MAX_CROSS_ENTROPY                        : float = 5.0
    
    # Perplexity value thresholds (actual perplexity values)
    PERPLEXITY_VERY_LOW_THRESHOLD            : float = 20.0
    PERPLEXITY_LOW_THRESHOLD                 : float = 40.0
    PERPLEXITY_HIGH_THRESHOLD                : float = 80.0
    PERPLEXITY_VERY_HIGH_THRESHOLD           : float = 150.0
    
    # Synthetic probability thresholds (normalized values 0-1)
    NORMALIZED_PERPLEXITY_HIGH_THRESHOLD     : float = 0.7
    NORMALIZED_PERPLEXITY_MEDIUM_THRESHOLD   : float = 0.5
    PERPLEXITY_VARIANCE_LOW_THRESHOLD        : float = 50.0
    PERPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 200.0
    STD_SENTENCE_PERPLEXITY_LOW_THRESHOLD    : float = 20.0
    STD_SENTENCE_PERPLEXITY_MEDIUM_THRESHOLD : float = 50.0
    CROSS_ENTROPY_LOW_THRESHOLD              : float = 0.3
    CROSS_ENTROPY_MEDIUM_THRESHOLD           : float = 0.6
    CHUNK_VARIANCE_VERY_LOW_THRESHOLD        : float = 25.0
    CHUNK_VARIANCE_LOW_THRESHOLD             : float = 100.0
    
    # Synthetic probability weights
    STRONG_SYNTHETIC_WEIGHT                  : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT                  : float = 0.6
    WEAK_SYNTHETIC_WEIGHT                    : float = 0.4
    VERY_WEAK_SYNTHETIC_WEIGHT               : float = 0.2
    VERY_LOW_SYNTHETIC_WEIGHT                : float = 0.3
    MINIMAL_SYNTHETIC_WEIGHT                 : float = 0.2
    
    # Confidence calculation
    CONFIDENCE_STD_NORMALIZER                : float = 0.5
    MIN_CONFIDENCE                           : float = 0.1
    MAX_CONFIDENCE                           : float = 0.9
    NEUTRAL_CONFIDENCE                       : float = 0.5
    LOW_FEATURE_CONFIDENCE                   : float = 0.3
    MIN_REQUIRED_FEATURES                    : int   = 3
    
    # Hybrid probability calculation
    NORMALIZED_PERPLEXITY_MIXED_MIN          : float = 0.4
    NORMALIZED_PERPLEXITY_MIXED_MAX          : float = 0.6
    PERPLEXITY_VARIANCE_HIGH_THRESHOLD       : float = 200.0
    PERPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 100.0
    STD_SENTENCE_PERPLEXITY_MIXED_MIN        : float = 20.0
    STD_SENTENCE_PERPLEXITY_MIXED_MAX        : float = 60.0
    MODERATE_HYBRID_WEIGHT                   : float = 0.4
    WEAK_HYBRID_WEIGHT                       : float = 0.2
    MINIMAL_HYBRID_WEIGHT                    : float = 0.0
    MAX_HYBRID_PROBABILITY                   : float = 0.3
    
    # Default feature values
    DEFAULT_OVERALL_PERPLEXITY               : float = 50.0
    DEFAULT_NORMALIZED_PERPLEXITY            : float = 0.5
    DEFAULT_AVG_SENTENCE_PERPLEXITY          : float = 50.0
    DEFAULT_STD_SENTENCE_PERPLEXITY          : float = 25.0
    DEFAULT_MIN_SENTENCE_PERPLEXITY          : float = 30.0
    DEFAULT_MAX_SENTENCE_PERPLEXITY          : float = 70.0
    DEFAULT_PERPLEXITY_VARIANCE              : float = 100.0
    DEFAULT_AVG_CHUNK_PERPLEXITY             : float = 50.0
    DEFAULT_CROSS_ENTROPY_SCORE              : float = 0.5
    
    # Math and normalization
    ZERO_TOLERANCE                           : float = 1e-10
    LARGE_PERPLEXITY_THRESHOLD               : float = 1000.0

    # Regular experssion for sentence splitting
    SENTENCE_SPLIT_PATTERN                   : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    

@dataclass(frozen = True)
class EntropyMetricParams:
    """
    Hyperparameters for Entropy Metric
    """
    # Text validation
    MIN_TEXT_LENGTH_FOR_ANALYSIS               : int   = 50
    MIN_SENTENCE_LENGTH                        : int   = 10
    MIN_WORDS_FOR_ANALYSIS                     : int   = 5
    MIN_TOKENS_FOR_ANALYSIS                    : int   = 10
    MIN_TOKENS_FOR_SEQUENCE                    : int   = 20
    
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB                 : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB                 : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT                : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START            : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START            : float = 0.7
    UNCERTAIN_RANGE_WIDTH                      : float = 0.4
    NEUTRAL_PROBABILITY                        : float = 0.5
    MIN_PROBABILITY                            : float = 0.0
    MAX_PROBABILITY                            : float = 1.0
    
    # Chunk analysis
    CHUNK_SIZE_WORDS                           : int   = 100
    CHUNK_OVERLAP_RATIO                        : float = 0.5
    MIN_CHUNK_LENGTH                           : int   = 20
    
    # Sequence analysis
    MAX_BIGRAM_ENTROPY                         : float = 8.0
    
    # Entropy normalization
    MAX_CHAR_ENTROPY                           : float = 4.0
    
    # Synthetic probability thresholds
    CHAR_ENTROPY_VERY_LOW_THRESHOLD            : float = 3.5
    CHAR_ENTROPY_LOW_THRESHOLD                 : float = 3.8
    CHAR_ENTROPY_MEDIUM_THRESHOLD              : float = 4.0
    ENTROPY_VARIANCE_VERY_LOW_THRESHOLD        : float = 0.1
    ENTROPY_VARIANCE_LOW_THRESHOLD             : float = 0.2
    ENTROPY_VARIANCE_MEDIUM_THRESHOLD          : float = 0.3
    TOKEN_DIVERSITY_LOW_THRESHOLD              : float = 0.6
    TOKEN_DIVERSITY_MEDIUM_THRESHOLD           : float = 0.7
    TOKEN_DIVERSITY_HIGH_THRESHOLD             : float = 0.8
    SEQUENCE_UNPREDICTABILITY_LOW_THRESHOLD    : float = 0.3
    SEQUENCE_UNPREDICTABILITY_MEDIUM_THRESHOLD : float = 0.4
    SEQUENCE_UNPREDICTABILITY_HIGH_THRESHOLD   : float = 0.5
    SYNTHETIC_PATTERN_SCORE_HIGH_THRESHOLD     : float = 0.75
    SYNTHETIC_PATTERN_SCORE_MEDIUM_THRESHOLD   : float = 0.5
    TOKEN_ENTROPY_LOW_THRESHOLD                : float = 6.5
    
    # Synthetic probability weights
    STRONG_SYNTHETIC_WEIGHT                    : float = 0.9
    VERY_STRONG_SYNTHETIC_WEIGHT               : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT                    : float = 0.7
    MODERATE_SYNTHETIC_WEIGHT                  : float = 0.6
    WEAK_SYNTHETIC_WEIGHT                      : float = 0.5
    VERY_WEAK_SYNTHETIC_WEIGHT                 : float = 0.4
    LOW_SYNTHETIC_WEIGHT                       : float = 0.3
    MINIMAL_SYNTHETIC_WEIGHT                   : float = 0.2
    VERY_LOW_SYNTHETIC_WEIGHT                  : float = 0.1
    
    # Confidence calculation
    CONFIDENCE_STD_NORMALIZER                  : float = 0.5
    MIN_CONFIDENCE                             : float = 0.1
    MAX_CONFIDENCE                             : float = 0.9
    NEUTRAL_CONFIDENCE                         : float = 0.5
    LOW_FEATURE_CONFIDENCE                     : float = 0.3
    MIN_REQUIRED_FEATURES                      : int   = 2
    
    # Hybrid probability calculation
    ENTROPY_VARIANCE_HIGH_THRESHOLD            : float = 0.5
    ENTROPY_VARIANCE_MIXED_THRESHOLD           : float = 0.3
    ENTROPY_DISCREPANCY_THRESHOLD              : float = 1.0
    SYNTHETIC_PATTERN_MIXED_MIN                : float = 0.4
    SYNTHETIC_PATTERN_MIXED_MAX                : float = 0.6
    STRONG_HYBRID_WEIGHT                       : float = 0.6
    MODERATE_HYBRID_WEIGHT                     : float = 0.4
    WEAK_HYBRID_WEIGHT                         : float = 0.3
    MINIMAL_HYBRID_WEIGHT                      : float = 0.0
    MAX_HYBRID_PROBABILITY                     : float = 0.4
      
    # Default feature values
    DEFAULT_CHAR_ENTROPY                       : float = 3.8
    DEFAULT_WORD_ENTROPY                       : float = 6.0
    DEFAULT_TOKEN_ENTROPY                      : float = 8.0
    DEFAULT_TOKEN_DIVERSITY                    : float = 0.7
    DEFAULT_SEQUENCE_UNPREDICTABILITY          : float = 0.5
    DEFAULT_ENTROPY_VARIANCE                   : float = 0.2
    DEFAULT_AVG_CHUNK_ENTROPY                  : float = 3.8
    DEFAULT_PREDICTABILITY_SCORE               : float = 0.5
    
    # Math and normalization
    ZERO_TOLERANCE                             : float = 1e-10


@dataclass(frozen = True)
class MultiPerturbationStabilityMetricParams:
    """
    Hyperparameters for Multi-Perturbation Stability Metric
    """
    # Text validation
    MIN_TEXT_LENGTH_FOR_ANALYSIS        : int   = 50
    MIN_TEXT_LENGTH_FOR_PERTURBATION    : int   = 10
    MIN_TOKENS_FOR_LIKELIHOOD           : int   = 3
    MIN_WORDS_FOR_PERTURBATION          : int   = 3
    MIN_WORDS_FOR_DELETION              : int   = 5
    
    # Domain threshold application - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB          : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB          : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT         : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START     : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START     : float = 0.7
    UNCERTAIN_RANGE_WIDTH               : float = 0.4
    NEUTRAL_PROBABILITY                 : float = 0.5
    MIN_PROBABILITY                     : float = 0.0
    MAX_PROBABILITY                     : float = 1.0
    
    # Perturbation parameters
    NUM_PERTURBATIONS                   : int   = 10
    MAX_PERTURBATION_ATTEMPTS           : int   = 10
    PERTURBATION_DELETION_RATIO         : float = 0.1
    ROBBERTA_TOP_K_PREDICTIONS          : int   = 3
    
    # Text preprocessing
    MAX_TEXT_LENGTH_FOR_ANALYSIS        : int   = 2000
    MAX_TEXT_LENGTH_FOR_PERTURBATION    : int   = 1000
    MAX_TOKEN_LENGTH                    : int   = 256
    MAX_ROBERTA_TOKEN_LENGTH            : int   = 128
    
    # Chunk analysis
    CHUNK_SIZE_WORDS                    : int   = 150
    CHUNK_OVERLAP_RATIO                 : float = 0.5
    MIN_CHUNK_LENGTH                    : int   = 50
    CHUNK_DELETION_RATIO                : float = 0.1
    
    # Likelihood calculation
    MIN_VALID_PERTURBATIONS             : int   = 3
    DEFAULT_LIKELIHOOD                  : float = 2.0
    MIN_LIKELIHOOD                      : float = 0.5
    MAX_LIKELIHOOD                      : float = 10.0
    
    # Stability scoring
    STABILITY_HIGH_THRESHOLD            : float = 0.7
    STABILITY_MEDIUM_THRESHOLD          : float = 0.5
    STABILITY_LOW_THRESHOLD             : float = 0.3
    RELATIVE_DROP_HIGH_THRESHOLD        : float = 0.5
    RELATIVE_DROP_MEDIUM_THRESHOLD      : float = 0.3
    RELATIVE_DROP_LOW_THRESHOLD         : float = 0.15
    
    # Curvature scoring
    CURVATURE_HIGH_THRESHOLD            : float = 0.7
    CURVATURE_MEDIUM_THRESHOLD          : float = 0.5
    CURVATURE_LOW_THRESHOLD             : float = 0.3
    CURVATURE_SCALING_FACTOR            : float = 3.0
    
    # Likelihood ratio thresholds
    LIKELIHOOD_RATIO_HIGH_THRESHOLD     : float = 0.8
    LIKELIHOOD_RATIO_MEDIUM_THRESHOLD   : float = 0.6
    LIKELIHOOD_RATIO_LOW_THRESHOLD      : float = 0.4
    MAX_LIKELIHOOD_RATIO                : float = 3.0
    MIN_LIKELIHOOD_RATIO                : float = 0.33
    
    # Stability variance thresholds
    STABILITY_VARIANCE_VERY_LOW         : float = 0.05
    STABILITY_VARIANCE_LOW              : float = 0.1
    STABILITY_VARIANCE_HIGH             : float = 0.15
    
    # Synthetic probability weights
    STABILITY_WEIGHT                    : float = 0.3
    CURVATURE_WEIGHT                    : float = 0.25
    RATIO_WEIGHT                        : float = 0.25
    VARIANCE_WEIGHT                     : float = 0.2
    
    # Synthetic probability thresholds
    STABILITY_STRONG_THRESHOLD          : float = 0.9
    STABILITY_MEDIUM_STRONG_THRESHOLD   : float = 0.7
    STABILITY_MODERATE_THRESHOLD        : float = 0.5
    STABILITY_WEAK_THRESHOLD            : float = 0.2
    CURVATURE_STRONG_THRESHOLD          : float = 0.8
    CURVATURE_MEDIUM_THRESHOLD          : float = 0.6
    CURVATURE_MODERATE_THRESHOLD        : float = 0.4
    CURVATURE_WEAK_THRESHOLD            : float = 0.2
    RATIO_STRONG_THRESHOLD              : float = 0.9
    RATIO_MEDIUM_THRESHOLD              : float = 0.7
    RATIO_MODERATE_THRESHOLD            : float = 0.5
    RATIO_WEAK_THRESHOLD                : float = 0.3
    VARIANCE_STRONG_THRESHOLD           : float = 0.8
    VARIANCE_MODERATE_THRESHOLD         : float = 0.5
    VARIANCE_WEAK_THRESHOLD             : float = 0.2
    
    # Confidence calculation
    CONFIDENCE_BASE                     : float = 0.5
    CONFIDENCE_STD_FACTOR               : float = 0.5
    MIN_CONFIDENCE                      : float = 0.1
    MAX_CONFIDENCE                      : float = 0.9
    NEUTRAL_CONFIDENCE                  : float = 0.5
    LOW_FEATURE_CONFIDENCE              : float = 0.3
    MIN_REQUIRED_FEATURES               : int   = 3
    
    # Hybrid probability calculation
    STABILITY_MIXED_MIN                 : float = 0.35
    STABILITY_MIXED_MAX                 : float = 0.55
    STABILITY_VARIANCE_MIXED_HIGH       : float = 0.15
    STABILITY_VARIANCE_MIXED_MEDIUM     : float = 0.1
    LIKELIHOOD_RATIO_MIXED_MIN          : float = 0.5
    LIKELIHOOD_RATIO_MIXED_MAX          : float = 0.8
    MODERATE_HYBRID_WEIGHT              : float = 0.4
    WEAK_HYBRID_WEIGHT                  : float = 0.3
    VERY_WEAK_HYBRID_WEIGHT             : float = 0.2
    MINIMAL_HYBRID_WEIGHT               : float = 0.0
    MAX_HYBRID_PROBABILITY              : float = 0.3
    
    # Default feature values
    DEFAULT_ORIGINAL_LIKELIHOOD         : float = 2.0
    DEFAULT_AVG_PERTURBED_LIKELIHOOD    : float = 1.8
    DEFAULT_LIKELIHOOD_RATIO            : float = 1.1
    DEFAULT_NORMALIZED_LIKELIHOOD_RATIO : float = 0.55
    DEFAULT_STABILITY_SCORE             : float = 0.3
    DEFAULT_CURVATURE_SCORE             : float = 0.3
    DEFAULT_PERTURBATION_VARIANCE       : float = 0.05
    DEFAULT_AVG_CHUNK_STABILITY         : float = 0.3
    DEFAULT_STABILITY_VARIANCE          : float = 0.1
    
    # Math and normalization
    ZERO_TOLERANCE                      : float = 1e-10
    
    # Common words to avoid masking
    COMMON_WORDS_TO_AVOID               : tuple = ('the', 'and', 'but', 'for', 'with', 'that', 'this', 'have', 'from', 'were')
    

@dataclass(frozen = True)
class MetricsEnsembleParams:
    """
    Constants for MEtrics Ensemble Classifier
    """
    # Minimum requirements 
    MIN_METRICS_REQUIRED             : int   = 3

    # Default probabilities
    DEFAULT_SYNTHETIC_PROB           : float = 0.5
    DEFAULT_AUTHENTIC_PROB           : float = 0.5
    DEFAULT_HYBRID_PROB              : float = 0.0

    # Weighting
    SIGMOID_CONFIDENCE_SCALE         : float = 10.0
    SIGMOID_CENTER                   : float = 0.5

    # Confidence composition
    CONFIDENCE_WEIGHT_BASE           : float = 0.4
    CONFIDENCE_WEIGHT_AGREEMENT      : float = 0.3
    CONFIDENCE_WEIGHT_CERTAINTY      : float = 0.2
    CONFIDENCE_WEIGHT_QUALITY        : float = 0.1

    # Uncertainty composition
    UNCERTAINTY_WEIGHT_VARIANCE      : float = 0.4
    UNCERTAINTY_WEIGHT_CONFIDENCE    : float = 0.3
    UNCERTAINTY_WEIGHT_DECISION      : float = 0.3

    # Consensus 
    CONSENSUS_STD_SCALING            : float = 2.0

    # Hybrid detection 
    HYBRID_PROB_THRESHOLD            : float = 0.25
    HYBRID_UNCERTAINTY_THRESHOLD     : float = 0.6
    HYBRID_SYNTHETIC_RANGE_LOW       : float = 0.3
    HYBRID_SYNTHETIC_RANGE_HIGH      : float = 0.7

    # Threshold adaptation 
    UNCERTAINTY_THRESHOLD_ADJUSTMENT : float = 0.1

    # Contribution labels
    CONTRIBUTION_HIGH                : float = 0.15
    CONTRIBUTION_MEDIUM              : float = 0.08

    HIGH_CONFIDENCE_THRESHOLD        : float = 0.7



# Singleton instances for parameter classes
document_extraction_params                 = DocumentExtractionParams()
language_detection_params                  = LanguageDetectionParams()
domain_classification_params               = DomainClassificationParams()
text_processing_params                     = TextProcessingParams()
base_metric_params                         = BaseMetricParams()
structural_metric_params                   = StructuralMetricParams()
semantic_analysis_params                   = SemanticAnalysisParams()
linguistic_metric_params                   = LinguisticMetricParams()
perplexity_metric_params                   = PerplexityMetricParams()
entropy_metric_params                      = EntropyMetricParams()
multi_perturbation_stability_metric_params = MultiPerturbationStabilityMetricParams()
metrics_ensemble_params                    = MetricsEnsembleParams()