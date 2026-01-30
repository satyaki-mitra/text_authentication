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
    MIN_CONFIDENCE_THRESHOLD        : float                = 0.20
    
    # Absolute Domain Confidence, below which everything will fallback to General Domain
    ABS_DOMAIN_CONFIDENCE_THRESHOLD : float                = 0.40
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD       : float                = 0.70
    MEDIUM_CONFIDENCE_THRESHOLD     : float                = 0.40
    LOW_CONFIDENCE_THRESHOLD        : float                = 0.25
    SECONDARY_DOMAIN_MIN_SCORE      : float                = 0.15
    
    # Mixed domain detection
    MIXED_DOMAIN_PRIMARY_MAX        : float                = 0.70
    MIXED_DOMAIN_SECONDARY_MIN      : float                = 0.30
    MIXED_DOMAIN_RATIO_THRESHOLD    : float                = 0.60
    MIXED_DOMAIN_CONFIDENCE_PENALTY : float                = 0.80
    
    # Text preprocessing
    MAX_WORDS_FOR_CLASSIFICATION    : int                  = 1000
    
    # Domain labels for zero-shot classification
    DOMAIN_LABELS                   : Dict[str, List[str]] = field(default_factory = lambda : {"academic"      : ["academic paper", "research article", "scientific paper", "scholarly writing", "thesis", "dissertation", "academic research"],
                                                                                               "creative"      : ["creative writing", "fiction", "story", "narrative", "poetry", "literary work", "imaginative writing"],
                                                                                               "ai_ml"         : ["artificial intelligence", "machine learning", "neural networks", "data science", "AI research", "deep learning", "AI", "GenAI", "Generative AI", "LLM", "Natural Langauge Processing", "NLP", "Statistics", "Bayesian"],
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
    DEFAULT_AUTHENTIC_PROBABILITY : float = 0.35
    DEFAULT_SYNTHETIC_PROBABILITY : float = 0.35
    DEFAULT_HYBRID_PROBABILITY    : float = 0.30
    DEFAULT_CONFIDENCE            : float = 0.0


@dataclass(frozen = True)
class StructuralMetricParams:
    """
    Hyperparameters for Structural Metric
    """
    # DOMAIN THRESHOLD APPLICATION - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB       : float = 0.65
    STRONG_AUTHENTIC_BASE_PROB       : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT      : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START  : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START  : float = 0.7
    UNCERTAIN_RANGE_WIDTH            : float = 0.4
    NEUTRAL_PROBABILITY              : float = 0.5
    MIN_PROBABILITY                  : float = 0.0
    MAX_PROBABILITY                  : float = 1.0
    
    # FEATURE EXTRACTION - TEXT PROCESSING PATTERNS
    SENTENCE_SPLIT_PATTERN           : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    WORD_TOKENIZE_PATTERN            : str   = r'\b\w+\b'
    PUNCTUATION_PATTERN              : str   = r'[^\w\s]'
    
    # BURSTINESS CALCULATION 
    # Burstiness = CV / normalization_factor
    # Empirical CV ranges      : 0.2-0.8 for most text
    # After /2.0 normalization : 0.1-0.4 typical range
    BURSTINESS_NORMALIZATION_FACTOR  : float = 2.0
    
    # CORRECTED thresholds based on empirical distributions
    BURSTINESS_LOW_THRESHOLD         : float = 0.15   # Below = very uniform (synthetic)
    BURSTINESS_MEDIUM_THRESHOLD      : float = 0.25   # Below = somewhat uniform
    BURSTINESS_HIGH_THRESHOLD        : float = 0.35   # Above = high variation (human/hybrid)
    
    # READABILITY CALCULATION (Flesch Reading Ease)
    FLESCH_CONSTANT_1                : float = 206.835
    FLESCH_CONSTANT_2                : float = 1.015
    FLESCH_CONSTANT_3                : float = 84.6
    NEUTRAL_READABILITY_SCORE        : float = 50.0
    MIN_READABILITY_SCORE            : float = 0.0
    MAX_READABILITY_SCORE            : float = 100.0
    
    # Readability thresholds for synthetic detection: Generative models often produce "optimal" readability (60-75)
    READABILITY_SYNTHETIC_MIN        : float = 60.0
    READABILITY_SYNTHETIC_MAX        : float = 75.0
    READABILITY_EXTREME_LOW          : float = 20.0   # Very difficult
    READABILITY_EXTREME_HIGH         : float = 90.0   # Very easy
    
    # REPETITION DETECTION
    REPETITION_WINDOW_SIZE           : int   = 10
    MIN_WORDS_FOR_REPETITION         : int   = 10
    REPETITION_LOW_THRESHOLD         : float = 0.1
    REPETITION_MEDIUM_THRESHOLD      : float = 0.2
    MIN_EXTREME_FEATURES             : int   = 2      
    
    # N-GRAM ANALYSIS
    BIGRAM_N                         : int   = 2
    TRIGRAM_N                        : int   = 3
    MIN_WORDS_FOR_NGRAM              : int   = 2
    
    # N-gram diversity thresholds
    # Lower diversity = more repetitive = potentially synthetic
    BIGRAM_DIVERSITY_LOW_THRESHOLD   : float = 0.7
    TRIGRAM_DIVERSITY_LOW_THRESHOLD  : float = 0.8
    
    # LENGTH UNIFORMITY THRESHOLDS
    # Length uniformity = 1 - (std / mean)
    # Higher uniformity = more consistent = potentially synthetic
    LENGTH_UNIFORMITY_HIGH_THRESHOLD : float = 0.7
    LENGTH_UNIFORMITY_MEDIUM_THRESH  : float = 0.5
    
    # SYNTHETIC PROBABILITY WEIGHTS
    STRONG_SYNTHETIC_WEIGHT          : float = 0.7
    MODERATE_SYNTHETIC_WEIGHT        : float = 0.5
    WEAK_SYNTHETIC_WEIGHT            : float = 0.4
    VERY_WEAK_SYNTHETIC_WEIGHT       : float = 0.3
    NEUTRAL_WEIGHT                   : float = 0.5
    
    # CONFIDENCE CALCULATION
    CONFIDENCE_BASE                  : float = 0.5    # Base confidence
    CONFIDENCE_STD_FACTOR            : float = 0.3    # Weight for agreement between indicators
    CONFIDENCE_SAMPLE_FACTOR         : float = 0.2    # Weight for sample size adequacy
    MIN_CONFIDENCE                   : float = 0.1
    MAX_CONFIDENCE                   : float = 0.9
    NEUTRAL_CONFIDENCE               : float = 0.5
    
    # Sample size thresholds for confidence
    MIN_SENTENCES_FOR_CONFIDENCE     : int   = 3      # Minimum sentences for reliable analysis
    MIN_WORDS_FOR_CONFIDENCE         : int   = 50     # Minimum words for reliable analysis
    CONFIDENCE_STD_NORMALIZER        : float = 0.5    # Kept for backward compatibility
    
    # HYBRID PROBABILITY CALCULATION
    SENTENCE_LENGTH_VARIANCE_RATIO   : float = 0.8
    TYPE_TOKEN_RATIO_EXTREME_LOW     : float = 0.3
    TYPE_TOKEN_RATIO_EXTREME_HIGH    : float = 0.9
    MODERATE_HYBRID_WEIGHT           : float = 0.4
    WEAK_HYBRID_WEIGHT               : float = 0.3
    MAX_HYBRID_PROBABILITY           : float = 0.4
    
    # FEATURE VALIDATION
    MIN_SENTENCE_LENGTH_FOR_STD      : int   = 2
    MIN_WORD_LENGTH_FOR_STD          : int   = 2
    MIN_VALUES_FOR_BURSTINESS        : int   = 2
    
    # MATH AND NORMALIZATION
    ZERO_TOLERANCE                   : float = 1e-10
    ZERO_VALUE                       : float = 0.0
    ONE_VALUE                        : float = 1.0


@dataclass(frozen = True)
class SemanticAnalysisParams:
    """
    Hyperparameters for Semantic Analysis Metric
    """
    # TEXT VALIDATION
    MIN_TEXT_LENGTH_FOR_ANALYSIS        : int   = 50
    MIN_SENTENCES_FOR_ANALYSIS          : int   = 3
    MIN_SENTENCE_LENGTH                 : int   = 10
    MIN_VALID_SENTENCE_LENGTH           : int   = 5
    
    # DOMAIN THRESHOLD APPLICATION
    STRONG_SYNTHETIC_BASE_PROB          : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB          : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT         : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START     : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START     : float = 0.7
    UNCERTAIN_RANGE_WIDTH               : float = 0.4
    NEUTRAL_PROBABILITY                 : float = 0.5
    MIN_PROBABILITY                     : float = 0.0
    MAX_PROBABILITY                     : float = 1.0
    
    # TEXT PROCESSING PATTERNS
    SENTENCE_SPLIT_PATTERN              : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    WORD_EXTRACTION_PATTERN             : str   = r'\b[a-zA-Z]{4,}\b'
    
    # COHERENCE CALCULATION: Coherence = average cosine similarity between adjacent sentences
    # Typical ranges:
    # - Very low (< 0.3)   : Incoherent, disconnected
    # - Low (0.3-0.5)      : Some connection
    # - Medium (0.5-0.7)   : Normal human writing
    # - High (0.7-0.85)    : Good flow, well-connected
    # - Very high (> 0.85) : Suspiciously perfect (potentially synthetic)
    
    SIMILARITY_VARIANCE_FACTOR          : float = 5.0       # Scaling factor for consistency calculation
    
    # COHERENCE THRESHOLDS : These define the "sweet spot" for human writing vs synthetic patterns
    COHERENCE_VERY_LOW_THRESHOLD        : float = 0.3       
    COHERENCE_LOW_THRESHOLD             : float = 0.5      
    COHERENCE_MEDIUM_LOW_THRESHOLD      : float = 0.65     
    COHERENCE_MEDIUM_HIGH_THRESHOLD     : float = 0.75     
    COHERENCE_HIGH_THRESHOLD            : float = 0.85     
    COHERENCE_SUSPICIOUS_THRESHOLD      : float = 0.9      
    
    # CONSISTENCY THRESHOLDS: Consistency = 1 - variance (lower variance = more consistent)
    CONSISTENCY_HIGH_THRESHOLD          : float = 0.8      
    CONSISTENCY_MEDIUM_THRESHOLD        : float = 0.6       
    CONSISTENCY_LOW_THRESHOLD           : float = 0.4      
    
    # REPETITION DETECTION
    REPETITION_SIMILARITY_THRESHOLD     : float = 0.8       
    REPETITION_SCORE_SCALING            : float = 3.0       
    MIN_SENTENCES_FOR_REPETITION        : int   = 5    
    
    REPETITION_HIGH_THRESHOLD           : float = 0.3     
    REPETITION_MEDIUM_THRESHOLD         : float = 0.15     
    REPETITION_LOW_THRESHOLD            : float = 0.05   
    
    # TOPIC DRIFT CALCULATION
    START_SECTION_SIZE                  : int   = 3        
    END_SECTION_SIZE                    : int   = 3       
    SECTION_SIZE_RATIO                  : int   = 3         
    
    TOPIC_DRIFT_LOW_THRESHOLD           : float = 0.2       
    TOPIC_DRIFT_MEDIUM_THRESHOLD        : float = 0.4       
    TOPIC_DRIFT_HIGH_THRESHOLD          : float = 0.6       
    
    # COHERENCE VARIANCE THRESHOLDS
    COHERENCE_VARIANCE_VERY_LOW         : float = 0.02   
    COHERENCE_VARIANCE_LOW_THRESHOLD    : float = 0.05      
    COHERENCE_VARIANCE_MEDIUM_THRESHOLD : float = 0.1       
    COHERENCE_VARIANCE_HIGH_THRESHOLD   : float = 0.15    
    
    # CHUNK ANALYSIS 
    CHUNK_SIZE_WORDS                    : int   = 200      
    CHUNK_OVERLAP_RATIO                 : float = 0.5      
    MIN_CHUNK_LENGTH                    : int   = 50       
    MIN_SENTENCES_PER_CHUNK             : int   = 2        
    
    # KEYWORD ANALYSIS
    MIN_WORDS_FOR_KEYWORD_ANALYSIS      : int   = 10       
    TOP_KEYWORDS_COUNT                  : int   = 10        
    MIN_KEYWORD_FREQUENCY               : int   = 2     
    
    # SYNTHETIC PROBABILITY WEIGHTS 
    COHERENCE_SUSPICIOUS_SYNTHETIC_WEIGHT : float = 0.8   
    COHERENCE_HIGH_SYNTHETIC_WEIGHT       : float = 0.6    
    COHERENCE_MEDIUM_SYNTHETIC_WEIGHT     : float = 0.4    
    COHERENCE_LOW_SYNTHETIC_WEIGHT        : float = 0.3  
    COHERENCE_INCOHERENT_SYNTHETIC_WEIGHT : float = 0.5     

    CONSISTENCY_STRONG_SYNTHETIC_WEIGHT   : float = 0.7   
    CONSISTENCY_MODERATE_SYNTHETIC_WEIGHT : float = 0.5     
    CONSISTENCY_WEAK_SYNTHETIC_WEIGHT     : float = 0.3    
    
    REPETITION_HIGH_SYNTHETIC_WEIGHT      : float = 0.6    
    REPETITION_MEDIUM_SYNTHETIC_WEIGHT    : float = 0.4     
    REPETITION_LOW_SYNTHETIC_WEIGHT       : float = 0.2  
    
    TOPIC_DRIFT_LOW_SYNTHETIC_WEIGHT      : float = 0.6     
    TOPIC_DRIFT_MEDIUM_SYNTHETIC_WEIGHT   : float = 0.4    
    TOPIC_DRIFT_HIGH_SYNTHETIC_WEIGHT     : float = 0.2     
    
    VARIANCE_LOW_SYNTHETIC_WEIGHT         : float = 0.6     
    VARIANCE_MEDIUM_SYNTHETIC_WEIGHT      : float = 0.4    
    VARIANCE_HIGH_SYNTHETIC_WEIGHT        : float = 0.2     
    
    # CONFIDENCE CALCULATION
    CONFIDENCE_BASE                       : float = 0.5     # Base confidence
    CONFIDENCE_STD_FACTOR                 : float = 0.3     # Weight for agreement between indicators
    CONFIDENCE_SAMPLE_FACTOR              : float = 0.2     # Weight for sample size adequacy
    CONFIDENCE_STD_NORMALIZER             : float = 0.5     # For backward compatibility
    MIN_CONFIDENCE                        : float = 0.1
    MAX_CONFIDENCE                        : float = 0.9
    NEUTRAL_CONFIDENCE                    : float = 0.5
    LOW_FEATURE_CONFIDENCE                : float = 0.3
    MIN_REQUIRED_FEATURES                 : int   = 3
    
    # Sample size thresholds for confidence
    MIN_SENTENCES_FOR_CONFIDENCE          : int   = 5       # Minimum sentences for reliable analysis
    MIN_CHUNKS_FOR_CONFIDENCE             : int   = 3       # Minimum chunks for reliable analysis
    
    # HYBRID PROBABILITY CALCULATION
    COHERENCE_MIXED_MIN                   : float = 0.55    
    COHERENCE_MIXED_MAX                   : float = 0.75   
    REPETITION_MIXED_MIN                  : float = 0.15    
    REPETITION_MIXED_MAX                  : float = 0.35   
    
    MODERATE_HYBRID_WEIGHT                : float = 0.4     
    WEAK_HYBRID_WEIGHT                    : float = 0.3    
    VERY_WEAK_HYBRID_WEIGHT               : float = 0.2    
    MAX_HYBRID_PROBABILITY                : float = 0.4     
    
    # DEFAULT FEATURE VALUES
    DEFAULT_COHERENCE                     : float = 0.5    
    DEFAULT_CONSISTENCY                   : float = 0.5   
    DEFAULT_REPETITION                    : float = 0.0   
    DEFAULT_TOPIC_DRIFT                   : float = 0.5  
    DEFAULT_CONTEXTUAL_CONSISTENCY        : float = 0.5   
    DEFAULT_CHUNK_COHERENCE               : float = 0.5  
    DEFAULT_COHERENCE_VARIANCE            : float = 0.1     
    
    # MATH AND NORMALIZATION
    ZERO_TOLERANCE                        : float = 1e-10   


@dataclass(frozen = True)
class LinguisticMetricParams:
    """
    Hyperparameters for Linguistic Metric
    """
    # TEXT VALIDATION
    MIN_TEXT_LENGTH_FOR_ANALYSIS             : int   = 50
    
    # DOMAIN THRESHOLD APPLICATION - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB               : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB               : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT              : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START          : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START          : float = 0.7
    UNCERTAIN_RANGE_WIDTH                    : float = 0.4
    NEUTRAL_PROBABILITY                      : float = 0.5
    MIN_PROBABILITY                          : float = 0.0
    MAX_PROBABILITY                          : float = 1.0
    
    # POS ANALYSIS
    MIN_TAGS_FOR_ENTROPY                     : int   = 10    

    # POS diversity thresholds (type-token ratio for POS tags): Normal English has diverse POS usage
    POS_DIVERSITY_LOW_THRESHOLD              : float = 0.3   
    POS_DIVERSITY_MEDIUM_THRESHOLD           : float = 0.5  
    POS_DIVERSITY_MIXED_MIN                  : float = 0.35  
    POS_DIVERSITY_MIXED_MAX                  : float = 0.55  
    
    # POS entropy thresholds: typical English POS entropy: 2.5-3.5 bits
    # Theoretical max for 17 POS tags: log2(17) ≈ 4.09 bits
    POS_ENTROPY_LOW_THRESHOLD                : float = 2.0   
    POS_ENTROPY_MEDIUM_THRESHOLD             : float = 2.8   
    POS_ENTROPY_HIGH_THRESHOLD               : float = 3.5   

    # SYNTACTIC COMPLEXITY
    COMPLEXITY_WEIGHT_AVG                    : float = 0.5   # Weight for average depth
    COMPLEXITY_WEIGHT_MAX                    : float = 0.5   # Weight for max depth
    
    # Syntactic complexity thresholds: Based on weighted combination of avg and max dependency depths
    # Typical range: 1.5-4.0
    SYNTACTIC_COMPLEXITY_LOW_THRESHOLD       : float = 2.0  
    SYNTACTIC_COMPLEXITY_MEDIUM_THRESHOLD    : float = 3.0  
    SYNTACTIC_COMPLEXITY_HIGH_THRESHOLD      : float = 4.0  
    
    # SENTENCE COMPLEXITY
    WORDS_PER_COMPLEXITY_UNIT                : float = 10.0  
    CLAUSE_COMPLEXITY_FACTOR                 : float = 0.5   
    CLAUSE_MARKERS                           : tuple = ('cc', 'mark') 
    
    # GRAMMATICAL PATTERNS
    TRANSITION_WORDS_SET                     : tuple = ('however', 'therefore', 'moreover', 'furthermore', 'consequently', 'additionally', 'nevertheless', 'nonetheless', 'thus', 'hence')
    IDEAL_PASSIVE_RATIO                      : float = 0.3   
    IDEAL_TRANSITION_RATIO                   : float = 0.2  
    PASSIVE_DEPENDENCY                       : str   = 'nsubjpass'  
    
    # Grammatical consistency thresholds
    GRAMMATICAL_CONSISTENCY_HIGH_THRESHOLD   : float = 0.8   
    GRAMMATICAL_CONSISTENCY_MEDIUM_THRESHOLD : float = 0.6  
    
    # Transition word usage thresholds
    TRANSITION_USAGE_HIGH_THRESHOLD          : float = 0.3   
    TRANSITION_USAGE_MEDIUM_THRESHOLD        : float = 0.15  
    
    # WRITING STYLE ANALYSIS
    IDEAL_LENGTH_VARIATION                   : float = 0.5  
    IDEAL_PUNCTUATION_RATIO                  : float = 0.1   
    
    # SYNTHETIC PATTERN DETECTION
    TRANSITION_OVERUSE_THRESHOLD             : float = 0.05  
    POS_SEQUENCE_FREQ_THRESHOLD              : float = 0.1  
    STRUCTURE_DIVERSITY_THRESHOLD            : float = 0.5   
    UNUSUAL_CONSTRUCTION_THRESHOLD           : float = 0.02 
    REPETITIVE_PHRASING_THRESHOLD            : float = 0.3  
    UNUSUAL_DEPENDENCIES                     : tuple = ('attr', 'oprd')  
    
    # Synthetic pattern score thresholds
    SYNTHETIC_PATTERN_HIGH_THRESHOLD         : float = 0.6   
    SYNTHETIC_PATTERN_MEDIUM_THRESHOLD       : float = 0.3   
    SYNTHETIC_PATTERN_MIXED_MIN              : float = 0.2  
    SYNTHETIC_PATTERN_MIXED_MAX              : float = 0.6  
    
    # CHUNK ANALYSIS
    CHUNK_SIZE_WORDS                         : int   = 200   
    CHUNK_OVERLAP_RATIO                      : float = 0.5  
    MIN_CHUNK_LENGTH                         : int   = 50    
    MIN_SENTENCES_FOR_STRUCTURE              : int   = 3     
    MIN_SENTENCES_FOR_ANALYSIS               : int   = 1     
    MIN_SENTENCES_FOR_CHUNK_VALIDITY         : int   = 1     
    
    # Complexity variance thresholds: Variance in syntactic complexity across chunks
    COMPLEXITY_VARIANCE_LOW_THRESHOLD        : float = 0.2  
    COMPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 0.5   
    COMPLEXITY_VARIANCE_HIGH_THRESHOLD       : float = 0.8  

    # SYNTHETIC PROBABILITY WEIGHTS
    STRONG_SYNTHETIC_WEIGHT                  : float = 0.9  
    MODERATE_SYNTHETIC_WEIGHT                : float = 0.8   
    MEDIUM_SYNTHETIC_WEIGHT                  : float = 0.7  
    WEAK_SYNTHETIC_WEIGHT                    : float = 0.6   
    VERY_WEAK_SYNTHETIC_WEIGHT               : float = 0.5   
    LOW_SYNTHETIC_WEIGHT                     : float = 0.4  
    VERY_LOW_SYNTHETIC_WEIGHT                : float = 0.3  
    MINIMAL_SYNTHETIC_WEIGHT                 : float = 0.2   
    
    # CONFIDENCE CALCULATION
    CONFIDENCE_BASE                          : float = 0.5    # Base confidence
    CONFIDENCE_STD_FACTOR                    : float = 0.3    # Weight for agreement between indicators
    CONFIDENCE_SAMPLE_FACTOR                 : float = 0.2    # Weight for sample size adequacy
    CONFIDENCE_STD_NORMALIZER                : float = 0.5    # For backward compatibility
    MIN_CONFIDENCE                           : float = 0.1
    MAX_CONFIDENCE                           : float = 0.9
    NEUTRAL_CONFIDENCE                       : float = 0.5
    LOW_FEATURE_CONFIDENCE                   : float = 0.3
    MIN_REQUIRED_FEATURES                    : int   = 4
    
    # Sample size thresholds for confidence
    MIN_SENTENCES_FOR_CONFIDENCE             : int   = 5     # Minimum sentences for reliable analysis
    MIN_CHUNKS_FOR_CONFIDENCE                : int   = 2     # Minimum chunks for reliable analysis
    
    # HYBRID PROBABILITY CALCULATION
    MODERATE_HYBRID_WEIGHT                   : float = 0.4   
    WEAK_HYBRID_WEIGHT                       : float = 0.3   
    MINIMAL_HYBRID_WEIGHT                    : float = 0.2   
    MAX_HYBRID_PROBABILITY                   : float = 0.4   
    
    # DEFAULT FEATURE VALUES
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
    DEFAULT_COMPLEXITY_VARIANCE              : float = 0.4   
    
    # MATH AND NORMALIZATION
    LOG_BASE                                 : int   = 2     
    ZERO_TOLERANCE                           : float = 1e-10 


@dataclass(frozen = True)
class PerplexityMetricParams:
    """
    Hyperparameters for Perplexity Metric
    """
    # TEXT VALIDATION
    MIN_TEXT_LENGTH_FOR_ANALYSIS             : int   = 50
    MIN_SENTENCE_LENGTH                      : int   = 20
    MIN_SENTENCE_LENGTH_DIVISOR              : int   = 2      # For min length checks (MIN_SENTENCE_LENGTH // 2)
    MIN_CHUNK_LENGTH                         : int   = 50
    MIN_CHUNK_SIZE_DIVISOR                   : int   = 2      
    
    # DOMAIN THRESHOLD APPLICATION - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB               : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB               : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT              : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START          : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START          : float = 0.7
    UNCERTAIN_RANGE_WIDTH                    : float = 0.4
    NEUTRAL_PROBABILITY                      : float = 0.5
    MIN_PROBABILITY                          : float = 0.0
    MAX_PROBABILITY                          : float = 1.0
    
    # MODEL PARAMETERS
    MAX_TOKEN_LENGTH                         : int   = 1024
    MIN_TOKENS_FOR_PERPLEXITY                : int   = 5
    
    # CHUNK ANALYSIS
    CHUNK_SIZE_WORDS                         : int   = 200
    CHUNK_OVERLAP_RATIO                      : float = 0.5
    
    # PERPLEXITY NORMALIZATION (Sigmoid Transformation)
    # normalized = 1 / (1 + exp((perplexity - center) / scale))
    # This maps perplexity values to [0, 1] range
    # Lower perplexity → higher normalized score → more synthetic-like
    PERPLEXITY_SIGMOID_CENTER                : float = 40.0    # Midpoint of sigmoid
    PERPLEXITY_SIGMOID_SCALE                 : float = 20.0    # Controls sigmoid steepness
    
    # CROSS-ENTROPY NORMALIZATION
    MAX_CROSS_ENTROPY                        : float = 5.0
    
    # PERPLEXITY VALUE THRESHOLDS (Actual Perplexity Values)
    # Typical perplexity ranges:
    # - Very low (< 20)   : Extremely predictable (likely synthetic)
    # - Low (20-40)       : Predictable (potentially synthetic)
    # - Medium (40-80)    : Moderate predictability
    # - High (80-150)     : Less predictable (likely human)
    # - Very high (> 150) : Highly unpredictable
    PERPLEXITY_VERY_LOW_THRESHOLD            : float = 20.0
    PERPLEXITY_LOW_THRESHOLD                 : float = 40.0
    PERPLEXITY_HIGH_THRESHOLD                : float = 80.0
    PERPLEXITY_VERY_HIGH_THRESHOLD           : float = 150.0
    
    # SYNTHETIC PROBABILITY THRESHOLDS (Normalized Values 0-1)
    # After sigmoid normalization:
    # - High normalized perplexity (> 0.7)     = low actual perplexity = synthetic
    # - Medium normalized perplexity (0.5-0.7) = uncertain
    # - Low normalized perplexity (< 0.5)      = high actual perplexity = authentic
    NORMALIZED_PERPLEXITY_HIGH_THRESHOLD     : float = 0.7
    NORMALIZED_PERPLEXITY_MEDIUM_THRESHOLD   : float = 0.5
    
    # Variance thresholds (low variance = consistent = synthetic)
    PERPLEXITY_VARIANCE_LOW_THRESHOLD        : float = 50.0
    PERPLEXITY_VARIANCE_MEDIUM_THRESHOLD     : float = 200.0
    PERPLEXITY_VARIANCE_HIGH_THRESHOLD       : float = 200.0   # For hybrid detection
    
    # Sentence perplexity standard deviation thresholds
    STD_SENTENCE_PERPLEXITY_LOW_THRESHOLD    : float = 20.0
    STD_SENTENCE_PERPLEXITY_MEDIUM_THRESHOLD : float = 50.0
    STD_SENTENCE_PERPLEXITY_MIXED_MIN        : float = 20.0
    STD_SENTENCE_PERPLEXITY_MIXED_MAX        : float = 60.0
    
    # Cross-entropy thresholds (lower = more predictable = synthetic)
    CROSS_ENTROPY_LOW_THRESHOLD              : float = 0.3
    CROSS_ENTROPY_MEDIUM_THRESHOLD           : float = 0.6
    
    # Chunk variance thresholds
    CHUNK_VARIANCE_VERY_LOW_THRESHOLD        : float = 25.0
    CHUNK_VARIANCE_LOW_THRESHOLD             : float = 100.0
    
    # SYNTHETIC PROBABILITY WEIGHTS
    STRONG_SYNTHETIC_WEIGHT                  : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT                  : float = 0.6
    WEAK_SYNTHETIC_WEIGHT                    : float = 0.4
    VERY_WEAK_SYNTHETIC_WEIGHT               : float = 0.2
    VERY_LOW_SYNTHETIC_WEIGHT                : float = 0.3
    MINIMAL_SYNTHETIC_WEIGHT                 : float = 0.2
    
    # CONFIDENCE CALCULATION 
    CONFIDENCE_BASE                          : float = 0.5     # Base confidence
    CONFIDENCE_STD_FACTOR                    : float = 0.3     # Weight for agreement between indicators
    CONFIDENCE_SAMPLE_FACTOR                 : float = 0.2     # Weight for sample size adequacy
    CONFIDENCE_STD_NORMALIZER                : float = 0.5     # For backward compatibility
    MIN_CONFIDENCE                           : float = 0.1
    MAX_CONFIDENCE                           : float = 0.9
    NEUTRAL_CONFIDENCE                       : float = 0.5
    LOW_FEATURE_CONFIDENCE                   : float = 0.3
    MIN_REQUIRED_FEATURES                    : int   = 3
    
    # Sample size thresholds for confidence
    MIN_SENTENCES_FOR_CONFIDENCE             : int   = 3      # NEW: Minimum sentences for reliable analysis
    MIN_CHUNKS_FOR_CONFIDENCE                : int   = 2      # NEW: Minimum chunks for reliable analysis
    
    # Moderate normalized perplexity suggests mixing
    NORMALIZED_PERPLEXITY_MIXED_MIN          : float = 0.4
    NORMALIZED_PERPLEXITY_MIXED_MAX          : float = 0.6
    
    # Hybrid probability weights
    MODERATE_HYBRID_WEIGHT                   : float = 0.4
    WEAK_HYBRID_WEIGHT                       : float = 0.2
    MINIMAL_HYBRID_WEIGHT                    : float = 0.0
    MAX_HYBRID_PROBABILITY                   : float = 0.4
    
    # These are used when analysis fails or as fallback values
    DEFAULT_OVERALL_PERPLEXITY               : float = 50.0    # Neutral perplexity
    DEFAULT_NORMALIZED_PERPLEXITY            : float = 0.5     # Neutral normalized value
    DEFAULT_AVG_SENTENCE_PERPLEXITY          : float = 50.0
    DEFAULT_STD_SENTENCE_PERPLEXITY          : float = 25.0
    DEFAULT_MIN_SENTENCE_PERPLEXITY          : float = 30.0
    DEFAULT_MAX_SENTENCE_PERPLEXITY          : float = 70.0
    DEFAULT_PERPLEXITY_VARIANCE              : float = 100.0
    DEFAULT_AVG_CHUNK_PERPLEXITY             : float = 50.0
    DEFAULT_CROSS_ENTROPY_SCORE              : float = 0.5
    
    # MATH AND NORMALIZATION
    ZERO_TOLERANCE                           : float = 1e-10
    LARGE_PERPLEXITY_THRESHOLD               : float = 1000.0  # Sanity check for unreasonably high values
    
    # TEXT PROCESSING
    SENTENCE_SPLIT_PATTERN                   : str   = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
    

@dataclass(frozen = True)
class EntropyMetricParams:
    """
    Hyperparameters for Entropy Metric
    """
    # TEXT VALIDATION
    MIN_TEXT_LENGTH_FOR_ANALYSIS               : int   = 50
    MIN_SENTENCE_LENGTH                        : int   = 10
    MIN_WORDS_FOR_ANALYSIS                     : int   = 5
    MIN_TOKENS_FOR_ANALYSIS                    : int   = 10
    MIN_TOKENS_FOR_SEQUENCE                    : int   = 20
    
    # DOMAIN THRESHOLD APPLICATION - PROBABILITY CONSTANTS
    STRONG_SYNTHETIC_BASE_PROB                 : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB                 : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT                : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START            : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START            : float = 0.7
    UNCERTAIN_RANGE_WIDTH                      : float = 0.4
    NEUTRAL_PROBABILITY                        : float = 0.5
    MIN_PROBABILITY                            : float = 0.0
    MAX_PROBABILITY                            : float = 1.0
    
    # CHUNK ANALYSIS
    CHUNK_SIZE_WORDS                           : int   = 100
    CHUNK_OVERLAP_RATIO                        : float = 0.5
    MIN_CHUNK_LENGTH                           : int   = 20
    
    # SEQUENCE ANALYSIS
    # Maximum theoretical bigram entropy for GPT-2 tokenizer (50257 tokens)
    # log2(50257^2) ≈ 31.2, but practical values are much lower (6-12)
    MAX_BIGRAM_ENTROPY                         : float = 12.0 

    # ENTROPY NORMALIZATION
    # Maximum character entropy for English (lowercase + space ≈ 27 chars)
    # Theoretical max: log2(27) ≈ 4.75
    # Practical max for natural text: 3.0-4.5
    MAX_CHAR_ENTROPY                           : float = 4.5  
    
    # CHARACTER ENTROPY THRESHOLDS 
    # Typical English text character entropy ranges: 3.0-4.5 bits
    # - Very low (< 3.0): Extremely repetitive (strong synthetic indicator)
    # - Low (3.0-3.5): Somewhat repetitive (moderate synthetic indicator)
    # - Medium (3.5-4.0): Normal variation
    # - High (> 4.0): High variation (authentic indicator)
    CHAR_ENTROPY_VERY_LOW_THRESHOLD            : float = 3.0    
    CHAR_ENTROPY_LOW_THRESHOLD                 : float = 3.5    
    CHAR_ENTROPY_MEDIUM_THRESHOLD              : float = 4.0    
    
    # ENTROPY VARIANCE THRESHOLDS
    # Variance in chunk entropies: Low variance = consistent = potentially synthetic
    ENTROPY_VARIANCE_VERY_LOW_THRESHOLD        : float = 0.05   
    ENTROPY_VARIANCE_LOW_THRESHOLD             : float = 0.15   
    ENTROPY_VARIANCE_MEDIUM_THRESHOLD          : float = 0.25   
    ENTROPY_VARIANCE_HIGH_THRESHOLD            : float = 0.4    
    ENTROPY_VARIANCE_MIXED_THRESHOLD           : float = 0.25   
    
    # TOKEN DIVERSITY THRESHOLDS
    # Type-token ratio (unique tokens / total tokens) depends heavily on text length:
    # - Short text (100 tokens): 0.7-0.9
    # - Medium text (500 tokens): 0.5-0.7  
    # - Long text (1000+ tokens): 0.3-0.5
    TOKEN_DIVERSITY_LOW_THRESHOLD              : float = 0.5    
    TOKEN_DIVERSITY_MEDIUM_THRESHOLD           : float = 0.65   
    TOKEN_DIVERSITY_HIGH_THRESHOLD             : float = 0.8    
    
    # SEQUENCE UNPREDICTABILITY THRESHOLDS
    # Normalized bigram entropy (0-1 scale after dividing by MAX_BIGRAM_ENTROPY): Lower = more predictable sequences = synthetic
    SEQUENCE_UNPREDICTABILITY_LOW_THRESHOLD    : float = 0.25   
    SEQUENCE_UNPREDICTABILITY_MEDIUM_THRESHOLD : float = 0.4    
    SEQUENCE_UNPREDICTABILITY_HIGH_THRESHOLD   : float = 0.6    
    
    # SYNTHETIC PATTERN SCORE THRESHOLDS
    # Proportion of synthetic patterns detected (0-1 scale)
    SYNTHETIC_PATTERN_SCORE_HIGH_THRESHOLD     : float = 0.75   
    SYNTHETIC_PATTERN_SCORE_MEDIUM_THRESHOLD   : float = 0.5    
    SYNTHETIC_PATTERN_MIXED_MIN                : float = 0.4    
    SYNTHETIC_PATTERN_MIXED_MAX                : float = 0.6    
    
    # TOKEN ENTROPY THRESHOLD
    # Token entropy typically ranges 6-10 for natural text
    # Lower = less diverse vocabulary = potentially synthetic
    TOKEN_ENTROPY_LOW_THRESHOLD                : float = 6.0    
    
    # SYNTHETIC PROBABILITY WEIGHTS
    STRONG_SYNTHETIC_WEIGHT                    : float = 0.9
    VERY_STRONG_SYNTHETIC_WEIGHT               : float = 0.8
    MEDIUM_SYNTHETIC_WEIGHT                    : float = 0.7
    MODERATE_SYNTHETIC_WEIGHT                  : float = 0.6
    WEAK_SYNTHETIC_WEIGHT                      : float = 0.5
    VERY_WEAK_SYNTHETIC_WEIGHT                 : float = 0.4
    LOW_SYNTHETIC_WEIGHT                       : float = 0.3
    MINIMAL_SYNTHETIC_WEIGHT                   : float = 0.2
    VERY_LOW_SYNTHETIC_WEIGHT                  : float = 0.1
    
    # CONFIDENCE CALCULATION 
    CONFIDENCE_BASE                            : float = 0.5    # Base confidence
    CONFIDENCE_STD_FACTOR                      : float = 0.3    # Weight for agreement between indicators
    CONFIDENCE_SAMPLE_FACTOR                   : float = 0.2    # Weight for sample size adequacy
    CONFIDENCE_STD_NORMALIZER                  : float = 0.5    # For backward compatibility
    MIN_CONFIDENCE                             : float = 0.1
    MAX_CONFIDENCE                             : float = 0.9
    NEUTRAL_CONFIDENCE                         : float = 0.5
    LOW_FEATURE_CONFIDENCE                     : float = 0.3
    MIN_REQUIRED_FEATURES                      : int   = 2
    
    # Sample size thresholds for confidence
    MIN_CHUNKS_FOR_CONFIDENCE                  : int   = 3     # Minimum chunks for reliable analysis
    MIN_TOKENS_FOR_CONFIDENCE                  : int   = 100   # Minimum tokens for reliable analysis
    
    # HYBRID PROBABILITY CALCULATION
    ENTROPY_DISCREPANCY_THRESHOLD              : float = 1.0    
    STRONG_HYBRID_WEIGHT                       : float = 0.6   
    MODERATE_HYBRID_WEIGHT                     : float = 0.4 
    WEAK_HYBRID_WEIGHT                         : float = 0.3    
    MINIMAL_HYBRID_WEIGHT                      : float = 0.0    
    MAX_HYBRID_PROBABILITY                     : float = 0.4    
    
    # DEFAULT FEATURE VALUES
    DEFAULT_CHAR_ENTROPY                       : float = 3.5    
    DEFAULT_WORD_ENTROPY                       : float = 6.0    
    DEFAULT_TOKEN_ENTROPY                      : float = 8.0    
    DEFAULT_TOKEN_DIVERSITY                    : float = 0.65  
    DEFAULT_SEQUENCE_UNPREDICTABILITY          : float = 0.5    
    DEFAULT_ENTROPY_VARIANCE                   : float = 0.2    
    DEFAULT_AVG_CHUNK_ENTROPY                  : float = 3.5  
    DEFAULT_PREDICTABILITY_SCORE               : float = 0.5    
    
    # MATH AND NORMALIZATION
    ZERO_TOLERANCE                             : float = 1e-10


@dataclass(frozen = True)
class MultiPerturbationStabilityMetricParams:
    """
    Hyperparameters for Multi-Perturbation Stability Metric: Based on statistical foundations and DetectGPT methodology
    """
    # TEXT VALIDATION
    MIN_TEXT_LENGTH_FOR_ANALYSIS        : int   = 50
    MIN_TEXT_LENGTH_FOR_PERTURBATION    : int   = 10
    MIN_TOKENS_FOR_LIKELIHOOD           : int   = 3
    MIN_WORDS_FOR_PERTURBATION          : int   = 3
    MIN_WORDS_FOR_DELETION              : int   = 5
    
    # DOMAIN THRESHOLD APPLICATION 
    STRONG_SYNTHETIC_BASE_PROB          : float = 0.7
    STRONG_AUTHENTIC_BASE_PROB          : float = 0.7
    WEAK_PROBABILITY_ADJUSTMENT         : float = 0.3
    UNCERTAIN_SYNTHETIC_RANGE_START     : float = 0.3
    UNCERTAIN_AUTHENTIC_RANGE_START     : float = 0.7
    UNCERTAIN_RANGE_WIDTH               : float = 0.4
    NEUTRAL_PROBABILITY                 : float = 0.5
    MIN_PROBABILITY                     : float = 0.0
    MAX_PROBABILITY                     : float = 1.0
    
    # PERTURBATION PARAMETERS
    NUM_PERTURBATIONS                   : int   = 20
    MAX_PERTURBATION_ATTEMPTS           : int   = 10
    PERTURBATION_DELETION_RATIO         : float = 0.13
    ROBBERTA_TOP_K_PREDICTIONS          : int   = 5
    
    # TEXT PREPROCESSING
    MAX_TEXT_LENGTH_FOR_ANALYSIS        : int   = 2000
    MAX_TEXT_LENGTH_FOR_PERTURBATION    : int   = 1000
    MAX_TOKEN_LENGTH                    : int   = 256
    MAX_ROBERTA_TOKEN_LENGTH            : int   = 128
    
    # CHUNK ANALYSIS
    CHUNK_SIZE_WORDS                    : int   = 150
    CHUNK_OVERLAP_RATIO                 : float = 0.5
    MIN_CHUNK_LENGTH                    : int   = 50
    CHUNK_DELETION_RATIO                : float = 0.1
    
    # These are NEGATIVE log-probabilities (cross-entropy loss values)
    MIN_VALID_PERTURBATIONS             : int   = 3
    DEFAULT_LOG_PROB                    : float = 5.0       # Typical negative log-prob for coherent text
    LOG_PROB_SANITY_MIN                 : float = 15.0      # Very incoherent text (high perplexity)
    LOG_PROB_SANITY_MAX                 : float = 1.0       # Very predictable text (low perplexity)
    
    # STABILITY SCORE CALCULATION 
    # Stability        = mean absolute difference between original and perturbed log-probs
    # Lower stability  = more synthetic (text remains predictable after perturbations)
    # Higher stability = more authentic (text becomes less predictable after perturbations)
    STABILITY_SYNTHETIC_THRESHOLD       : float = 0.5       # Below this = likely synthetic
    STABILITY_AUTHENTIC_THRESHOLD       : float = 1.5       # Above this = likely authentic
    STABILITY_SCALING_FACTOR            : float = 1.0       # For normalization if needed
    
    # CURVATURE SCORE CALCULATION 
    # Curvature      = variance of log-prob differences across perturbations
    # Low curvature  = smooth likelihood surface = more synthetic
    # High curvature = rough likelihood surface = more authentic
    CURVATURE_SYNTHETIC_THRESHOLD       : float = 0.1       # Below this = likely synthetic
    CURVATURE_AUTHENTIC_THRESHOLD       : float = 0.5       # Above this = likely authentic
    CURVATURE_SCALING_FACTOR            : float = 2.0       # Variance is typically small, scale for interpretability
    
    # STABILITY VARIANCE THRESHOLDS (For chunk consistency analysis)
    STABILITY_VARIANCE_VERY_LOW         : float = 0.05      # Very consistent = synthetic
    STABILITY_VARIANCE_LOW              : float = 0.1       # Somewhat consistent
    STABILITY_VARIANCE_MEDIUM           : float = 0.2       # Moderate variance
    STABILITY_VARIANCE_HIGH             : float = 0.3       # High variance = authentic
    
    # FEATURE WEIGHTS
    STABILITY_WEIGHT                    : float = 0.45      # Primary signal (most reliable)
    CURVATURE_WEIGHT                    : float = 0.35      # Secondary signal (surface smoothness)
    VARIANCE_WEIGHT                     : float = 0.20      # Tertiary signal (consistency check)
    
    # For stability score interpretation
    STABILITY_STRONG_SYNTHETIC          : float = 0.3       # Very low stability
    STABILITY_MODERATE_SYNTHETIC        : float = 0.8       # Medium stability
    STABILITY_WEAK_SYNTHETIC            : float = 1.2       # Higher stability
    STABILITY_AUTHENTIC                 : float = 1.8       # Very high stability
    
    # For curvature score interpretation
    CURVATURE_STRONG_SYNTHETIC          : float = 0.05      # Very low curvature
    CURVATURE_MODERATE_SYNTHETIC        : float = 0.2       # Medium curvature
    CURVATURE_WEAK_SYNTHETIC            : float = 0.4       # Higher curvature
    CURVATURE_AUTHENTIC                 : float = 0.7       # Very high curvature
    
    # For variance interpretation
    VARIANCE_STRONG_SYNTHETIC           : float = 0.05      # Very low variance
    VARIANCE_MODERATE_SYNTHETIC         : float = 0.15      # Medium variance
    VARIANCE_WEAK_SYNTHETIC             : float = 0.25      # Higher variance
    VARIANCE_AUTHENTIC                  : float = 0.35      # Very high variance
    
    # Probability weights for different levels
    PROB_WEIGHT_STRONG                  : float = 0.9       # High confidence synthetic
    PROB_WEIGHT_MODERATE                : float = 0.7       # Medium confidence synthetic
    PROB_WEIGHT_WEAK                    : float = 0.5       # Low confidence synthetic
    PROB_WEIGHT_NEUTRAL                 : float = 0.3       # Uncertain
    PROB_WEIGHT_AUTHENTIC               : float = 0.1       # Likely authentic
    
    # CONFIDENCE CALCULATION
    CONFIDENCE_BASE                     : float = 0.5       # Base confidence
    CONFIDENCE_PERTURBATION_FACTOR      : float = 0.3       # More valid perturbations = higher confidence
    CONFIDENCE_AGREEMENT_FACTOR         : float = 0.2       # Agreement between signals = higher confidence
    MIN_CONFIDENCE                      : float = 0.1       # Minimum reportable confidence
    MAX_CONFIDENCE                      : float = 0.9       # Maximum reportable confidence
    NEUTRAL_CONFIDENCE                  : float = 0.5       # Neutral confidence level
    LOW_FEATURE_CONFIDENCE              : float = 0.3       # Low confidence when features insufficient
    MIN_REQUIRED_FEATURES               : int   = 3         # Minimum features needed for confident assessment
    
    # HYBRID PROBABILITY CALCULATION
    STABILITY_MIXED_MIN                 : float = 0.5       # Lower bound for mixed content stability
    STABILITY_MIXED_MAX                 : float = 1.0       # Upper bound for mixed content stability
    CURVATURE_MIXED_MIN                 : float = 0.2       # Lower bound for mixed content curvature
    CURVATURE_MIXED_MAX                 : float = 0.4       # Upper bound for mixed content curvature
    VARIANCE_MIXED_MIN                  : float = 0.1       # Lower bound for mixed content variance
    VARIANCE_MIXED_MAX                  : float = 0.25      # Upper bound for mixed content variance
    
    MODERATE_HYBRID_WEIGHT              : float = 0.4       # Strong hybrid indicator
    WEAK_HYBRID_WEIGHT                  : float = 0.3       # Moderate hybrid indicator
    VERY_WEAK_HYBRID_WEIGHT             : float = 0.2       # Weak hybrid indicator
    MINIMAL_HYBRID_WEIGHT               : float = 0.0       # No hybrid indication
    MAX_HYBRID_PROBABILITY              : float = 0.4       # Maximum hybrid probability
    
    # DEFAULT FEATURE VALUES
    DEFAULT_ORIGINAL_LOG_PROB           : float = 5.0       # Neutral log-probability
    DEFAULT_AVG_PERTURBED_LOG_PROB      : float = 5.5       # Slightly higher (less predictable after perturbation)
    DEFAULT_STABILITY_SCORE             : float = 0.8       # Neutral stability
    DEFAULT_CURVATURE_SCORE             : float = 0.3       # Neutral curvature
    DEFAULT_PERTURBATION_VARIANCE       : float = 0.2       # Neutral variance
    DEFAULT_AVG_CHUNK_STABILITY         : float = 0.8       # Neutral chunk stability
    DEFAULT_STABILITY_VARIANCE          : float = 0.2       # Neutral stability variance
    
    # MATH AND NORMALIZATION
    ZERO_TOLERANCE                      : float = 1e-10     # Numerical stability threshold
    
    # COMMON WORDS TO AVOID MASKING
    COMMON_WORDS_TO_AVOID               : tuple = ('the', 'and', 'but', 'for', 'with', 'that', 'this', 'have', 'from', 'were',
                                                   'been', 'being', 'very', 'most', 'more', 'some', 'such', 'into', 'also',
                                                   'than', 'them', 'they', 'their', 'there', 'these', 'those', 'what', 'when',
                                                   'where', 'which', 'while', 'will', 'would', 'could', 'should')

    

@dataclass(frozen = True)
class MetricsEnsembleParams:
    """
    Constants for Metrics Ensemble Classifier
    """
    # MINIMUM REQUIREMENTS 
    MIN_METRICS_REQUIRED                  : int   = 3

    # DEFAULT PROBABILITIES (for fallback/error cases)
    DEFAULT_SYNTHETIC_PROB                : float = 0.5
    DEFAULT_AUTHENTIC_PROB                : float = 0.5
    DEFAULT_HYBRID_PROB                   : float = 0.0

    CALIBRATION_TEMP_MIN                  : float = 1.0
    CALIBRATION_TEMP_MAX                  : float = 3.0

    # SIGMOID CONFIDENCE ADJUSTMENT
    # Formula: sigmoid(scale * (confidence - center))
    # This creates a non-linear weighting where:
    # - Low confidence metrics get heavily downweighted
    # - High confidence metrics get upweighted
    # - Center point (0.5) is the inflection point
    SIGMOID_CONFIDENCE_SCALE              : float = 8.0    # Steepness of sigmoid
    SIGMOID_CENTER                        : float = 0.5    # Center of sigmoid

    # CALIBRATION PARAMETERS: Since we can't properly apply temperature scaling to probabilities, we use Platt scaling instead (beta distribution calibration)
    PLATT_SCALING_ALPHA                   : float = 1.3    # Shape parameter (> 1 = sharpen, < 1 = soften)
    PLATT_SCALING_BETA                    : float = 1.3    # Shape parameter
    USE_PLATT_SCALING                     : bool  = True   # Enable/disable calibration
    
    # Alternative: Simple power scaling (prob^exponent)
    POWER_CALIBRATION_EXPONENT            : float = 0.85   # < 1 softens probabilities
    USE_POWER_CALIBRATION                 : bool  = False  # Alternative to Platt
    
    # CONFIDENCE LABELING THRESHOLDS
    # These define confidence levels based on distance from 0.5
    # "Very High": prob < 0.10 or > 0.90
    # "High": prob < 0.20 or > 0.80
    # "Moderate": prob < 0.30 or > 0.70
    # "Low": everything else
    CONFIDENCE_VERY_HIGH_BOUNDARY         : float = 0.10   
    CONFIDENCE_HIGH_BOUNDARY              : float = 0.20   
    CONFIDENCE_MODERATE_BOUNDARY          : float = 0.30   
    
    # DECISION PARAMETERS
    MAX_CONFIDENCE                        : float = 1.0
    MAX_DECISION_UNCERTAINTY              : float = 1.0
    DECISION_UNCERTAINTY_SCALE            : float = 2.0    # Amplifies distance from center
    DECISION_AMBIGUITY_CENTER             : float = 0.5    # Center point for ambiguity
    DECISION_MARGIN                       : float = 0.05   # Safety margin for decisions

    # UNCERTAINTY COMPOSITION
    # Uncertainty = weighted combination of:
    # 1. Variance across metric predictions
    # 2. Average confidence of metrics
    # 3. Closeness to decision boundary (0.5)
    UNCERTAINTY_WEIGHT_VARIANCE           : float = 0.4
    UNCERTAINTY_WEIGHT_CONFIDENCE         : float = 0.3
    UNCERTAINTY_WEIGHT_DECISION           : float = 0.3

    # CONFIDENCE COMPOSITION: Overall confidence = weighted combination of:
    # 1. Weighted average of individual metric confidences
    # 2. Agreement/consensus among metrics
    CONFIDENCE_WEIGHT_EVIDENCE            : float = 0.70
    CONFIDENCE_WEIGHT_CONSENSUS           : float = 0.30

    # CONSENSUS CALCULATION
    # Consensus = 1 - (std_dev * scale)
    # Lower standard deviation = higher consensus
    CONSENSUS_STD_SCALING                 : float = 2.0
    METRICS_DISAGREEMENT_THRESHOLD_HIGH   : float = 0.7    # High uncertainty warning
    METRICS_DISAGREEMENT_THRESHOLD_STRONG : float = 0.8    # Strong consensus indicator

    # HYBRID DETECTION 
    HYBRID_PROB_THRESHOLD                 : float = 0.20   # Direct hybrid probability threshold
    HYBRID_UNCERTAINTY_THRESHOLD          : float = 0.55   # Uncertainty level suggesting mixed content
    HYBRID_SYNTHETIC_RANGE_LOW            : float = 0.35   # Lower bound of "mixed zone"
    HYBRID_SYNTHETIC_RANGE_HIGH           : float = 0.65   # Upper bound of "mixed zone"

    # THRESHOLD ADAPTATION : Adjust decision threshold based on uncertainty
    # Higher uncertainty requires higher confidence for classification
    UNCERTAINTY_THRESHOLD_ADJUSTMENT      : float = 0.10

    # ABSTENTION / COVERAGE CONTROL: Selective prediction: Don't make a decision if confidence is too low
    MIN_CONFIDENCE_FOR_DECISION           : float = 0.50   # Minimum overall confidence
    MAX_UNCERTAINTY_FOR_DECISION          : float = 0.55   # Maximum acceptable uncertainty
    MIN_CONSENSUS_FOR_DECISION            : float = 0.40   # Minimum metric consensus


@dataclass(frozen = True)
class OrchestrationParameters:
    """
    Constants for Orchestration Layer with Long text handling
    """
    # Text Limit for Domain Classification
    MAX_WORDS_FOR_CLASSIFICATION        : int   = 500

    # Windowing 
    MAX_SINGLE_ANALYSIS_WORDS           : int   = 500   # Process texts under 800 words normally
    WINDOW_SIZE_WORDS                   : int   = 400   # Each window size
    WINDOW_OVERLAP_WORDS                : int   = 150   # Overlap between windows
    WINDOW_LOW_VARIANCE_THRESHOLD       : float = 0.03
    MIN_VALID_METRICS_RATIO_PER_WINDOW  : float = 0.5

    # Decision logic
    WINDOW_VARIANCE_CONSENSUS_SCALE     : float = 2.0
    MIN_WINDOW_WORDS_ABSOLUTE           : int   = 200
    WINDOW_VERDICT_MARGIN               : float = 0.12
    WINDOW_VERDICT_CONFIDENCE_GATE      : float = 0.60

    # Stability thresholds
    STABILITY_HARD_OVERRIDE             : float = 0.25
    STABILITY_HARD_MIN_SYNTHETIC        : float = 0.65 # floor on synthetic prob
    STABILITY_HARD_CONFIDENCE_BOOST     : float = 0.10
    STABILITY_HARD_CONFIDENCE_CAP       : float = 0.80
    HIGH_VARIANCE_CONFIDENCE_MULTIPLIER : float = 0.85



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
orchestration_parameters                   = OrchestrationParameters()