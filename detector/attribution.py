# DEPENDENCIES
import re
import numpy as np
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple 
from loguru import logger
from typing import Optional 
from dataclasses import dataclass
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from processors.text_processor import ProcessedText



class AIModel(Enum):
    """
    Supported AI models for attribution - ALIGNED WITH DOCUMENTATION
    """
    GPT_3_5         = "gpt-3.5-turbo"
    GPT_4           = "gpt-4"
    GPT_4_TURBO     = "gpt-4-turbo"
    GPT_4o          = "gpt-4o"
    CLAUDE_3_OPUS   = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU  = "claude-3-haiku"
    GEMINI_PRO      = "gemini-pro"
    GEMINI_ULTRA    = "gemini-ultra"
    GEMINI_FLASH    = "gemini-flash"
    LLAMA_2         = "llama-2"
    LLAMA_3         = "llama-3"
    MISTRAL         = "mistral"
    MIXTRAL         = "mixtral"
    DEEPSEEK_CHAT   = "deepseek-chat"
    DEEPSEEK_CODER  = "deepseek-coder"
    HUMAN           = "human"
    UNKNOWN         = "unknown"


@dataclass
class AttributionResult:
    """
    Result of AI model attribution
    """
    predicted_model     : AIModel
    confidence          : float
    model_probabilities : Dict[str, float]
    reasoning           : List[str]
    fingerprint_matches : Dict[str, int]
    domain_used         : Domain
    metric_contributions: Dict[str, float]
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"predicted_model"     : self.predicted_model.value,
                "confidence"          : round(self.confidence, 4),
                "model_probabilities" : {model: round(prob, 4) for model, prob in self.model_probabilities.items()},
                "reasoning"           : self.reasoning,
                "fingerprint_matches" : self.fingerprint_matches,
                "domain_used"         : self.domain_used.value,
                "metric_contributions": {metric: round(contrib, 4) for metric, contrib in self.metric_contributions.items()},
               }


class ModelAttributor:
    """
    Model attribution
    
    FEATURES:
    - Domain-aware calibration
    - 6-metric ensemble integration  
    - Confidence-weighted aggregation
    - Explainable reasoning
    """
    # DOCUMENT-ALIGNED: Metric weights from technical specification
    METRIC_WEIGHTS           = {"perplexity"                   : 0.25,  
                                "structural"                   : 0.15,   
                                "semantic_analysis"            : 0.15,  
                                "entropy"                      : 0.20,  
                                "linguistic"                   : 0.15,  
                                "multi_perturbation_stability" : 0.10,  
                               }
    
    # DOMAIN-AWARE model patterns for ALL 16 DOMAINS
    DOMAIN_MODEL_PREFERENCES = {Domain.GENERAL       : [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO, AIModel.GPT_3_5],
                                Domain.ACADEMIC      : [AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.GEMINI_ULTRA, AIModel.GPT_4_TURBO],
                                Domain.TECHNICAL_DOC : [AIModel.GPT_4_TURBO, AIModel.CLAUDE_3_SONNET, AIModel.LLAMA_3, AIModel.GPT_4],
                                Domain.AI_ML         : [AIModel.GPT_4_TURBO, AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.DEEPSEEK_CODER],
                                Domain.SOFTWARE_DEV  : [AIModel.GPT_4_TURBO, AIModel.DEEPSEEK_CODER, AIModel.CLAUDE_3_SONNET, AIModel.GPT_4],
                                Domain.ENGINEERING   : [AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.GPT_4_TURBO, AIModel.LLAMA_3],
                                Domain.SCIENCE       : [AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.GEMINI_ULTRA, AIModel.GPT_4_TURBO],
                                Domain.BUSINESS      : [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO, AIModel.GPT_3_5],
                                Domain.LEGAL         : [AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.GPT_4_TURBO, AIModel.CLAUDE_3_SONNET],
                                Domain.MEDICAL       : [AIModel.GPT_4, AIModel.CLAUDE_3_OPUS, AIModel.GEMINI_ULTRA, AIModel.GPT_4_TURBO],
                                Domain.JOURNALISM    : [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO, AIModel.GPT_3_5],
                                Domain.CREATIVE      : [AIModel.CLAUDE_3_OPUS, AIModel.GPT_4, AIModel.GEMINI_PRO, AIModel.CLAUDE_3_SONNET],
                                Domain.MARKETING     : [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO, AIModel.GPT_3_5],
                                Domain.SOCIAL_MEDIA  : [AIModel.GPT_3_5, AIModel.GEMINI_PRO, AIModel.DEEPSEEK_CHAT, AIModel.LLAMA_3],
                                Domain.BLOG_PERSONAL : [AIModel.CLAUDE_3_SONNET, AIModel.GPT_4, AIModel.GEMINI_PRO, AIModel.GPT_3_5],
                                Domain.TUTORIAL      : [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET, AIModel.GEMINI_PRO, AIModel.GPT_4_TURBO],
                               }

    # Enhanced Model-specific fingerprints with comprehensive patterns
    MODEL_FINGERPRINTS = {AIModel.GPT_3_5       : {"phrases"              : ["as an ai language model",
                                                                             "i don't have personal opinions",
                                                                             "it's important to note that",
                                                                             "it's worth noting that", 
                                                                             "keep in mind that",
                                                                             "bear in mind that",
                                                                             "i should point out",
                                                                             "it's also important to",
                                                                             "additionally, it's worth",
                                                                             "furthermore, it should be",
                                                                             "i cannot provide",
                                                                             "i'm unable to",
                                                                             "i don't have the ability",
                                                                             "based on the information",
                                                                             "according to the context",
                                                                            ],
                                                   "sentence_starters"    : ["however,",
                                                                             "additionally,",
                                                                             "furthermore,",
                                                                             "moreover,",
                                                                             "in conclusion,",
                                                                             "therefore,",
                                                                             "consequently,",
                                                                             "as a result,",
                                                                             "in summary,",
                                                                             "ultimately,",
                                                                            ],
                                                   "structural_patterns"  : ["firstly", 
                                                                             "secondly", 
                                                                             "thirdly",
                                                                             "on one hand", 
                                                                             "on the other hand",
                                                                             "in terms of", 
                                                                             "with regard to",
                                                                            ],
                                                   "punctuation_patterns" : {"em_dash_frequency"     : (0.01, 0.03),
                                                                             "semicolon_frequency"   : (0.005, 0.015),
                                                                             "parentheses_frequency" : (0.01, 0.04),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length"     : (18, 25),
                                                                             "transition_word_density" : (0.08, 0.15),
                                                                             "formality_score"         : (0.7, 0.9),
                                                                             "hedging_language"        : (0.05, 0.12),
                                                                            }
                                                  },
                          AIModel.GPT_4         : {"phrases"              : ["it's important to note that",
                                                                             "it's worth mentioning that",
                                                                             "to clarify this point",
                                                                             "in other words,",
                                                                             "that being said,",
                                                                             "in essence,",
                                                                             "fundamentally,",
                                                                             "at its core,",
                                                                             "from a broader perspective",
                                                                             "when considering",
                                                                             "this suggests that",
                                                                             "this implies that",
                                                                             "it follows that",
                                                                             "consequently,",
                                                                             "accordingly,",
                                                                            ],
                                                   "sentence_starters"    : ["interestingly,",
                                                                             "notably,",
                                                                             "crucially,",
                                                                             "essentially,",
                                                                             "ultimately,",
                                                                             "significantly,",
                                                                             "importantly,",
                                                                             "remarkably,",
                                                                             "surprisingly,",
                                                                            ],
                                                   "structural_patterns"  : ["in light of", 
                                                                             "with respect to", 
                                                                             "pertaining to",
                                                                             "as evidenced by", 
                                                                             "as indicated by", 
                                                                             "as suggested by",
                                                                            ],
                                                   "punctuation_patterns" : {"em_dash_frequency"  : (0.02, 0.05),
                                                                             "colon_frequency"     : (0.01, 0.03),
                                                                             "semicolon_frequency" : (0.01, 0.02),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length"       : (20, 28),
                                                                             "vocabulary_sophistication" : (0.7, 0.9),
                                                                             "conceptual_density"        : (0.6, 0.85),
                                                                             "analytical_depth"          : (0.65, 0.9),
                                                                            }
                                                  },
                          AIModel.CLAUDE_3_OPUS : {"phrases"              : ["i'd be glad to",
                                                                             "i'm happy to help",
                                                                             "let me explain this",
                                                                             "to clarify this further",
                                                                             "in this context,",
                                                                             "from this perspective,",
                                                                             "building on that point",
                                                                             "expanding on this idea",
                                                                             "delving deeper into",
                                                                             "to elaborate further",
                                                                             "it's worth considering",
                                                                             "this raises the question",
                                                                             "this highlights the importance",
                                                                             "this underscores the need",
                                                                            ],
                                                   "sentence_starters"    : ["certainly,",
                                                                             "indeed,",
                                                                             "particularly,",
                                                                             "specifically,",
                                                                             "notably,",
                                                                             "importantly,",
                                                                             "interestingly,",
                                                                             "crucially,",
                                                                            ],
                                                   "structural_patterns"  : ["in other words", 
                                                                             "to put it differently", 
                                                                             "that is to say",
                                                                             "for instance",
                                                                             "for example", 
                                                                             "as an illustration",
                                                                            ],
                                                   "punctuation_patterns" : {"em_dash_frequency"   : (0.015, 0.04),
                                                                             "parenthetical_usage" : (0.02, 0.06),
                                                                             "colon_frequency"     : (0.008, 0.025),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length" : (17, 24),
                                                                             "nuanced_language"    : (0.6, 0.85),
                                                                             "explanatory_depth"   : (0.7, 0.95),
                                                                             "conceptual_clarity"  : (0.65, 0.9),
                                                                            }
                                                  },
                          AIModel.GEMINI_PRO    : {"phrases"              : ["here's what you need to know",
                                                                             "here's how it works",
                                                                             "let's explore this",
                                                                             "let's look at this",
                                                                             "consider this example",
                                                                             "think of it this way",
                                                                             "imagine if you will",
                                                                             "picture this scenario",
                                                                             "to break it down",
                                                                             "in simple terms",
                                                                             "put simply,",
                                                                             "basically,",
                                                                             "the key point is",
                                                                             "the main idea here",
                                                                            ],
                                                   "sentence_starters"    : ["now,",
                                                                             "so,",
                                                                             "well,",
                                                                             "basically,",
                                                                             "essentially,",
                                                                             "actually,",
                                                                             "technically,",
                                                                             "practically,",
                                                                            ],
                                                   "structural_patterns"  : ["on that note", 
                                                                             "speaking of which", 
                                                                             "by the way",
                                                                             "as a side note", 
                                                                             "incidentally", 
                                                                             "in any case",
                                                                            ],
                                                   "punctuation_patterns" : {"exclamation_frequency" : (0.01, 0.03),
                                                                             "question_frequency"    : (0.02, 0.05),
                                                                             "ellipsis_frequency"    : (0.005, 0.02),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length" : (15, 22),
                                                                             "conversational_tone" : (0.5, 0.8),
                                                                             "accessibility_score" : (0.6, 0.9),
                                                                             "engagement_level"    : (0.55, 0.85),
                                                                            }
                                                  },
                          AIModel.LLAMA_3       : {"phrases"              : ["it's worth noting",
                                                                             "it's important to understand",
                                                                             "this means that",
                                                                             "this indicates that",
                                                                             "this shows that",
                                                                             "this demonstrates that",
                                                                             "based on this,",
                                                                             "given this context",
                                                                             "in this case,",
                                                                             "for this reason",
                                                                             "as such,",
                                                                             "therefore,",
                                                                            ],
                                                   "sentence_starters"    : ["first,",
                                                                             "second,",
                                                                             "third,",
                                                                             "next,",
                                                                             "then,",
                                                                             "finally,",
                                                                             "overall,",
                                                                             "in general,",
                                                                            ],
                                                   "structural_patterns"  : ["in addition", 
                                                                             "moreover",
                                                                             "furthermore",
                                                                             "however", 
                                                                             "nevertheless", 
                                                                             "nonetheless",
                                                                            ],
                                                   "punctuation_patterns" : {"comma_frequency"       : (0.08, 0.15),
                                                                             "period_frequency"      : (0.06, 0.12),
                                                                             "conjunction_frequency" : (0.05, 0.1),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length"    : (16, 23),
                                                                             "directness_score"       : (0.6, 0.85),
                                                                             "clarity_score"          : (0.65, 0.9),
                                                                             "structural_consistency" : (0.7, 0.95),
                                                                            }
                                                  },
                          AIModel.DEEPSEEK_CHAT : {"phrases"              : ["i understand you're asking",
                                                                             "let me help you with that",
                                                                             "i can assist you with",
                                                                             "regarding your question",
                                                                             "to answer your question",
                                                                             "in response to your query",
                                                                             "based on your request",
                                                                             "as per your question",
                                                                             "concerning your inquiry",
                                                                             "with respect to your question",
                                                                             "i'll do my best to",
                                                                             "i'll try to help you",
                                                                             "allow me to explain",
                                                                             "let me break it down",
                                                                            ],
                                                   "sentence_starters"    : ["well,",
                                                                             "okay,",
                                                                             "so,",
                                                                             "now,",
                                                                             "first,",
                                                                             "actually,",
                                                                             "specifically,",
                                                                             "generally,",
                                                                            ],
                                                   "structural_patterns"  : ["in other words", 
                                                                             "to put it simply", 
                                                                             "that is",
                                                                             "for example",
                                                                             "for instance", 
                                                                             "such as",
                                                                            ],
                                                   "punctuation_patterns" : {"comma_frequency"    : (0.07, 0.14),
                                                                             "period_frequency"   : (0.05, 0.11),
                                                                             "question_frequency" : (0.01, 0.04),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length" : (14, 21),
                                                                             "helpfulness_tone"    : (0.6, 0.9),
                                                                             "explanatory_style"   : (0.55, 0.85),
                                                                             "user_focus"          : (0.65, 0.95),
                                                                            }
                                                  },
                          AIModel.MIXTRAL       : {"phrases"              : ["it should be noted that",
                                                                             "it is important to recognize",
                                                                             "this suggests that",
                                                                             "this implies that",
                                                                             "this indicates that",
                                                                             "from this we can see",
                                                                             "based on this analysis",
                                                                             "considering these points",
                                                                             "taking into account",
                                                                             "in light of these factors",
                                                                            ],
                                                   "sentence_starters"    : ["however,",
                                                                             "moreover,",
                                                                             "furthermore,",
                                                                             "additionally,",
                                                                             "conversely,",
                                                                             "similarly,",
                                                                             "likewise,",
                                                                            ],
                                                   "structural_patterns"  : ["on the one hand", 
                                                                             "on the other hand",
                                                                             "in contrast", 
                                                                             "by comparison",
                                                                             "as opposed to",
                                                                             "rather than",
                                                                            ],
                                                   "punctuation_patterns" : {"semicolon_frequency"   : (0.008, 0.02),
                                                                             "colon_frequency"       : (0.006, 0.018),
                                                                             "parentheses_frequency" : (0.012, 0.035),
                                                                            },
                                                   "style_markers"        : {"avg_sentence_length"  : (19, 26),
                                                                             "analytical_tone"      : (0.65, 0.9),
                                                                             "comparative_language" : (0.5, 0.8),
                                                                             "balanced_perspective" : (0.6, 0.85),
                                                                            }
                                                  }
                         }

    
    def __init__(self):
        """
        Initialize model attributor with domain awareness
        """
        self.is_initialized = False
        logger.info("ModelAttributor initialized with domain-aware calibration")
    

    def initialize(self) -> bool:
        """
        Initialize attribution system
        """
        try:
            self.is_initialized = True
            logger.success("Model attribution system initialized with metric ensemble")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize attribution system: {repr(e)}")
            return False
    

    def attribute(self, text: str, processed_text: Optional[ProcessedText] = None, metric_results: Optional[Dict[str, MetricResult]] = None,
                  domain: Domain = Domain.GENERAL) -> AttributionResult:
        """
        Attribute text to specific AI model with domain awareness
        
        Arguments:
        ----------
            text           { str }           : Input text

            processed_text { ProcessedText } : Processed text metadata
            
            metric_results { dict }          : Results from 6 core metrics
            
            domain         { Domain }        : Text domain for calibration
            
        Returns:
        --------
            { AttributionResult }            : Attribution result with domain context
        """
        try:
            # Get domain-specific model preferences
            domain_preferences                    = self.DOMAIN_MODEL_PREFERENCES.get(domain, [AIModel.GPT_4, AIModel.CLAUDE_3_SONNET])
            
            # Fingerprint analysis
            fingerprint_scores                    = self._calculate_fingerprint_scores(text   = text,
                                                                                       domain = domain,
                                                                                      )
            
            # Statistical pattern analysis
            statistical_scores                    = self._analyze_statistical_patterns(text   = text, 
                                                                                       domain = domain,
                                                                                      )
            
            # Metric-based attribution using all 6 metrics
            metric_scores                         = self._analyze_metric_patterns(metric_results = metric_results, domain = domain) if metric_results else {}
            
            # Ensemble Combination
            combined_scores, metric_contributions = self._combine_attribution_scores(fingerprint_scores = fingerprint_scores,
                                                                                     statistical_scores = statistical_scores,
                                                                                     metric_scores      = metric_scores,
                                                                                     domain             = domain,
                                                                                    )
            
            # Domain-aware prediction - FIXED: Always show the actual highest probability model
            predicted_model, confidence           = self._make_domain_aware_prediction(combined_scores    = combined_scores,
                                                                                       domain             = domain,
                                                                                       domain_preferences = domain_preferences,
                                                                                      )
            
            # Reasoning with domain context - FIXED
            reasoning                             = self._generate_detailed_reasoning(predicted_model      = predicted_model,
                                                                                      confidence           = confidence,
                                                                                      domain               = domain,
                                                                                      metric_contributions = metric_contributions,
                                                                                      combined_scores      = combined_scores,
                                                                                     )
            
            return AttributionResult(predicted_model      = predicted_model,
                                     confidence           = confidence,
                                     model_probabilities  = combined_scores,
                                     reasoning            = reasoning,
                                     fingerprint_matches  = self._get_top_fingerprints(fingerprint_scores),
                                     domain_used          = domain,
                                     metric_contributions = metric_contributions,
                                    )
            
        except Exception as e:
            logger.error(f"Error in model attribution: {repr(e)}")
            return self._create_unknown_result(domain)


    def _calculate_fingerprint_scores(self, text: str, domain: Domain) -> Dict[AIModel, float]:
        """
        Calculate fingerprint match scores with DOMAIN CALIBRATION - FIXED for all domains
        """
        scores             = {model: 0.0 for model in AIModel if model not in [AIModel.HUMAN, AIModel.UNKNOWN]}
        
        # Adjust sensitivity based on all domains
        domain_sensitivity = {Domain.GENERAL       : 1.00,   
                              Domain.ACADEMIC      : 1.20,   
                              Domain.CREATIVE      : 0.90,   
                              Domain.AI_ML         : 1.15,  
                              Domain.SOFTWARE_DEV  : 1.15,
                              Domain.TECHNICAL_DOC : 1.10,   
                              Domain.ENGINEERING   : 1.10,   
                              Domain.SCIENCE       : 1.20,   
                              Domain.BUSINESS      : 1.05,  
                              Domain.LEGAL         : 1.25,  
                              Domain.MEDICAL       : 1.20,   
                              Domain.JOURNALISM    : 1.00,  
                              Domain.MARKETING     : 0.95, 
                              Domain.SOCIAL_MEDIA  : 0.80,   
                              Domain.BLOG_PERSONAL : 0.90,   
                              Domain.TUTORIAL      : 1.00,  
                             }
        
        sensitivity        = domain_sensitivity.get(domain, 1.0)
        text_lower         = text.lower()
        
        for model, fingerprints in self.MODEL_FINGERPRINTS.items():
            match_count  = 0
            total_checks = 0
            
            # Check phrase matches
            if ("phrases" in fingerprints):
                for phrase in fingerprints["phrases"]:
                    if (phrase in text_lower):
                        match_count += 3
                    
                    total_checks += 1
            
            # Check sentence starters
            if ("sentence_starters" in fingerprints):
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    sentence = sentence.strip().lower()
                    for starter in fingerprints["sentence_starters"]:
                        if (sentence.startswith(starter)):
                            match_count += 2
                            break
                
                total_checks += len(sentences)
            
            # Check structural patterns
            if ("structural_patterns" in fingerprints):
                for pattern in fingerprints["structural_patterns"]:
                    if (pattern in text_lower):
                        match_count += 2
                    
                    total_checks += 1
            
            # Calculate normalized score
            if (total_checks > 0):
                base_score    = min(1.0, match_count / (total_checks * 0.5))
                # Apply domain calibration
                scores[model] = min(1.0, base_score * sensitivity)
        
        return scores


    def _analyze_statistical_patterns(self, text: str, domain: Domain) -> Dict[AIModel, float]:
        """
        Analyze statistical patterns to identify model with domain awareness
        """
        scores    = {model: 0.3 for model in AIModel if model not in [AIModel.HUMAN, AIModel.UNKNOWN]}
        
        # Calculate text statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words     = text.split()
        
        if not sentences or not words:
            return scores
        
        # Basic statistics
        avg_sentence_length = len(words) / len(sentences)
        word_count          = len(words)
        sentence_count      = len(sentences)
        
        # Punctuation frequencies
        em_dash_freq        = text.count('—') / word_count if word_count else 0
        semicolon_freq      = text.count(';') / word_count if word_count else 0
        colon_freq          = text.count(':') / word_count if word_count else 0
        comma_freq          = text.count(',') / word_count if word_count else 0
        question_freq       = text.count('?') / sentence_count if sentence_count else 0
        exclamation_freq    = text.count('!') / sentence_count if sentence_count else 0
        
        # DOMAIN-AWARE: Adjust expectations based on domains
        domain_adjustments  = {Domain.GENERAL       : 1.00,
                               Domain.ACADEMIC      : 1.10,   
                               Domain.CREATIVE      : 0.95, 
                               Domain.AI_ML         : 1.05,  
                               Domain.SOFTWARE_DEV  : 1.05, 
                               Domain.TECHNICAL_DOC : 1.05, 
                               Domain.ENGINEERING   : 1.05, 
                               Domain.SCIENCE       : 1.08,  
                               Domain.BUSINESS      : 1.00,  
                               Domain.LEGAL         : 1.12,  
                               Domain.MEDICAL       : 1.08,  
                               Domain.JOURNALISM    : 0.95,  
                               Domain.MARKETING     : 0.92,  
                               Domain.SOCIAL_MEDIA  : 0.85,  
                               Domain.BLOG_PERSONAL : 0.95, 
                               Domain.TUTORIAL      : 1.00, 
                              }
        
        domain_factor       = domain_adjustments.get(domain, 1.0)
        
        # Compare against model fingerprints
        for model, fingerprints in self.MODEL_FINGERPRINTS.items():
            if ("style_markers" not in fingerprints) or ("punctuation_patterns" not in fingerprints):
                continue
            
            style       = fingerprints["style_markers"]
            punct       = fingerprints["punctuation_patterns"]
            match_score = 0.3
            
            # Check sentence length with domain adjustment
            if ("avg_sentence_length" in style):
                min_len, max_len = style["avg_sentence_length"]
                adjusted_min     = min_len * domain_factor
                adjusted_max     = max_len * domain_factor
                
                if (adjusted_min <= avg_sentence_length <= adjusted_max):
                    match_score += 0.25
            
            # Check punctuation patterns
            punctuation_checks = [("em_dash_frequency", em_dash_freq),
                                  ("semicolon_frequency", semicolon_freq),
                                  ("colon_frequency", colon_freq),
                                  ("comma_frequency", comma_freq),
                                  ("question_frequency", question_freq),
                                  ("exclamation_frequency", exclamation_freq),
                                 ]
            
            for pattern_name, observed_freq in punctuation_checks:
                if (pattern_name in punct):
                    min_freq, max_freq = punct[pattern_name]

                    if (min_freq <= observed_freq <= max_freq):
                        match_score += 0.08
            
            scores[model] = min(1.0, match_score)
        
        return scores


    def _analyze_metric_patterns(self, metric_results: Dict[str, MetricResult], domain: Domain) -> Dict[AIModel, float]:
        """
        Use all 6 metrics with proper weights for attribution
        """
        scores                = {model: 0.0 for model in AIModel if model not in [AIModel.HUMAN, AIModel.UNKNOWN]}
        
        if not metric_results:
            return scores
        
        # DOMAIN-AWARE: Adjust metric sensitivity based on domain 
        domain_metric_weights = {Domain.GENERAL       : {"perplexity": 1.0, "structural": 1.0, "entropy": 1.0, "semantic_analysis": 1.0, "linguistic": 1.0, "multi_perturbation_stability": 1.0},
                                 Domain.ACADEMIC      : {"perplexity": 1.2, "structural": 1.0, "entropy": 0.9, "semantic_analysis": 1.1, "linguistic": 1.3, "multi_perturbation_stability": 0.8},
                                 Domain.TECHNICAL_DOC : {"perplexity": 1.2, "structural": 1.1, "entropy": 0.9, "semantic_analysis": 1.2, "linguistic": 1.1, "multi_perturbation_stability": 0.8},
                                 Domain.AI_ML         : {"perplexity": 1.3, "structural": 1.0, "entropy": 0.9, "semantic_analysis": 1.2, "linguistic": 1.2, "multi_perturbation_stability": 0.8},
                                 Domain.SOFTWARE_DEV  : {"perplexity": 1.2, "structural": 1.1, "entropy": 0.9, "semantic_analysis": 1.1, "linguistic": 1.0, "multi_perturbation_stability": 0.9},
                                 Domain.ENGINEERING   : {"perplexity": 1.2, "structural": 1.1, "entropy": 0.9, "semantic_analysis": 1.1, "linguistic": 1.2, "multi_perturbation_stability": 0.8},
                                 Domain.SCIENCE       : {"perplexity": 1.2, "structural": 1.0, "entropy": 0.9, "semantic_analysis": 1.2, "linguistic": 1.3, "multi_perturbation_stability": 0.8},
                                 Domain.BUSINESS      : {"perplexity": 1.1, "structural": 1.0, "entropy": 1.0, "semantic_analysis": 1.2, "linguistic": 1.1, "multi_perturbation_stability": 0.9},
                                 Domain.LEGAL         : {"perplexity": 1.2, "structural": 1.1, "entropy": 0.9, "semantic_analysis": 1.3, "linguistic": 1.3, "multi_perturbation_stability": 0.8},
                                 Domain.MEDICAL       : {"perplexity": 1.2, "structural": 1.0, "entropy": 0.9, "semantic_analysis": 1.2, "linguistic": 1.2, "multi_perturbation_stability": 0.8},
                                 Domain.JOURNALISM    : {"perplexity": 1.1, "structural": 1.0, "entropy": 1.0, "semantic_analysis": 1.1, "linguistic": 1.1, "multi_perturbation_stability": 0.9},
                                 Domain.CREATIVE      : {"perplexity": 0.9, "structural": 0.9, "entropy": 1.2, "semantic_analysis": 1.0, "linguistic": 1.3, "multi_perturbation_stability": 0.9},
                                 Domain.MARKETING     : {"perplexity": 1.0, "structural": 1.0, "entropy": 1.1, "semantic_analysis": 1.1, "linguistic": 1.2, "multi_perturbation_stability": 0.8},
                                 Domain.SOCIAL_MEDIA  : {"perplexity": 1.0, "structural": 0.8, "entropy": 1.3, "semantic_analysis": 0.9, "linguistic": 0.9, "multi_perturbation_stability": 0.9},
                                 Domain.BLOG_PERSONAL : {"perplexity": 1.0, "structural": 0.9, "entropy": 1.2, "semantic_analysis": 1.0, "linguistic": 1.1, "multi_perturbation_stability": 0.8},
                                 Domain.TUTORIAL      : {"perplexity": 1.1, "structural": 1.0, "entropy": 1.0, "semantic_analysis": 1.1, "linguistic": 1.1, "multi_perturbation_stability": 0.9},
                                }
        
        domain_weights        = domain_metric_weights.get(domain, domain_metric_weights[Domain.GENERAL])
        
        # PERPLEXITY ANALYSIS (25% weight)
        if ("perplexity" in metric_results):
            perplexity_result  = metric_results["perplexity"]
            overall_perplexity = perplexity_result.details.get("overall_perplexity", 50)
            domain_weight      = domain_weights.get("perplexity", 1.0)
            
            # GPT models typically have lower perplexity
            if (overall_perplexity < 25):
                scores[AIModel.GPT_4]       += 0.6 * self.METRIC_WEIGHTS["perplexity"] * domain_weight
                scores[AIModel.GPT_4_TURBO] += 0.5 * self.METRIC_WEIGHTS["perplexity"] * domain_weight

            elif (overall_perplexity < 35):
                scores[AIModel.GPT_3_5]    += 0.4 * self.METRIC_WEIGHTS["perplexity"] * domain_weight
                scores[AIModel.GEMINI_PRO] += 0.3 * self.METRIC_WEIGHTS["perplexity"] * domain_weight
        
        # STRUCTURAL ANALYSIS (15% weight)
        if ("structural" in metric_results):
            structural_result = metric_results["structural"]
            burstiness        = structural_result.details.get("burstiness_score", 0.5)
            uniformity        = structural_result.details.get("length_uniformity", 0.5)
            domain_weight     = domain_weights.get("structural", 1.0)
            
            # Claude models show more structural consistency
            if (uniformity > 0.7):
                scores[AIModel.CLAUDE_3_OPUS]   += 0.5 * self.METRIC_WEIGHTS["structural"] * domain_weight
                scores[AIModel.CLAUDE_3_SONNET] += 0.4 * self.METRIC_WEIGHTS["structural"] * domain_weight
        
        # SEMANTIC ANALYSIS (15% weight)
        if ("semantic_analysis" in metric_results):
            semantic_result = metric_results["semantic_analysis"]
            coherence       = semantic_result.details.get("coherence_score", 0.5)
            consistency     = semantic_result.details.get("consistency_score", 0.5)
            domain_weight   = domain_weights.get("semantic_analysis", 1.0)
            
            # GPT-4 shows exceptional semantic coherence
            if (coherence > 0.8):
                scores[AIModel.GPT_4]       += 0.7 * self.METRIC_WEIGHTS["semantic_analysis"] * domain_weight
                scores[AIModel.GPT_4_TURBO] += 0.6 * self.METRIC_WEIGHTS["semantic_analysis"] * domain_weight
        
        # ENTROPY ANALYSIS (20% weight)
        if ("entropy" in metric_results):
            entropy_result            = metric_results["entropy"]
            token_diversity           = entropy_result.details.get("token_diversity", 0.5)
            sequence_unpredictability = entropy_result.details.get("sequence_unpredictability", 0.5)
            domain_weight             = domain_weights.get("entropy", 1.0)
            
            # Higher entropy diversity suggests more sophisticated models
            if (token_diversity > 0.7):
                scores[AIModel.CLAUDE_3_OPUS] += 0.6 * self.METRIC_WEIGHTS["entropy"] * domain_weight
                scores[AIModel.GPT_4]         += 0.5 * self.METRIC_WEIGHTS["entropy"] * domain_weight
        
        # LINGUISTIC ANALYSIS (15% weight)
        if ("linguistic" in metric_results):
            linguistic_result    = metric_results["linguistic"]
            pos_diversity        = linguistic_result.details.get("pos_diversity", 0.5)
            syntactic_complexity = linguistic_result.details.get("syntactic_complexity", 2.5)
            domain_weight        = domain_weights.get("linguistic", 1.0)
            
            # Complex linguistic patterns suggest advanced models
            if (syntactic_complexity > 3.0):
                scores[AIModel.CLAUDE_3_OPUS] += 0.5 * self.METRIC_WEIGHTS["linguistic"] * domain_weight
                scores[AIModel.GPT_4]         += 0.4 * self.METRIC_WEIGHTS["linguistic"] * domain_weight
        
        # MULTI-PERTURBATION STABILITY ANALYSIS (10% weight)
        if ("multi_perturbation_stability" in metric_results):
            multi_perturbation_stability_result = metric_results["multi_perturbation_stability"]
            stability                           = multi_perturbation_stability_result.details.get("stability_score", 0.5)
            curvature                           = multi_perturbation_stability_result.details.get("curvature_score", 0.5)
            
            # Specific stability patterns for different model families
            if (0.4 <= stability <= 0.6):
                scores[AIModel.MIXTRAL] += 0.4 * self.METRIC_WEIGHTS["multi_perturbation_stability"]
                scores[AIModel.LLAMA_3] += 0.3 * self.METRIC_WEIGHTS["multi_perturbation_stability"]
        
        # Normalize scores
        for model in scores:
            scores[model] = min(1.0, scores[model])
        
        return scores


    def _combine_attribution_scores(self, fingerprint_scores: Dict[AIModel, float], statistical_scores: Dict[AIModel, float],
                                    metric_scores: Dict[AIModel, float], domain: Domain) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        ENSEMBLE COMBINATION using document-specified weights and domain awareness
        """
        # DOMAIN-AWARE weighting for ALL 16 DOMAINS
        domain_weights       = {Domain.GENERAL       : {"fingerprint": 0.35, "statistical": 0.30, "metric": 0.35},
                                Domain.ACADEMIC      : {"fingerprint": 0.30, "statistical": 0.35, "metric": 0.35},
                                Domain.TECHNICAL_DOC : {"fingerprint": 0.25, "statistical": 0.40, "metric": 0.35},
                                Domain.AI_ML         : {"fingerprint": 0.28, "statistical": 0.37, "metric": 0.35},
                                Domain.SOFTWARE_DEV  : {"fingerprint": 0.27, "statistical": 0.38, "metric": 0.35},
                                Domain.ENGINEERING   : {"fingerprint": 0.28, "statistical": 0.37, "metric": 0.35},
                                Domain.SCIENCE       : {"fingerprint": 0.30, "statistical": 0.35, "metric": 0.35},
                                Domain.BUSINESS      : {"fingerprint": 0.33, "statistical": 0.35, "metric": 0.32},
                                Domain.LEGAL         : {"fingerprint": 0.28, "statistical": 0.40, "metric": 0.32},
                                Domain.MEDICAL       : {"fingerprint": 0.30, "statistical": 0.38, "metric": 0.32},
                                Domain.JOURNALISM    : {"fingerprint": 0.35, "statistical": 0.33, "metric": 0.32},
                                Domain.CREATIVE      : {"fingerprint": 0.40, "statistical": 0.30, "metric": 0.30},
                                Domain.MARKETING     : {"fingerprint": 0.38, "statistical": 0.32, "metric": 0.30},
                                Domain.SOCIAL_MEDIA  : {"fingerprint": 0.45, "statistical": 0.35, "metric": 0.20},
                                Domain.BLOG_PERSONAL : {"fingerprint": 0.42, "statistical": 0.32, "metric": 0.26},
                                Domain.TUTORIAL      : {"fingerprint": 0.33, "statistical": 0.35, "metric": 0.32},
                               }
        
        weights              = domain_weights.get(domain, domain_weights[Domain.GENERAL])
        
        combined             = dict()
        metric_contributions = dict()
        
        all_models           = set(fingerprint_scores.keys()) | set(statistical_scores.keys()) | set(metric_scores.keys())
        
        for model in all_models:
            score                 = (fingerprint_scores.get(model, 0.0) * weights["fingerprint"] + 
                                     statistical_scores.get(model, 0.0) * weights["statistical"] + 
                                     metric_scores.get(model, 0.0) * weights["metric"]
                                    )
            
            combined[model.value] = score
        
        # Normalize scores to sum to 1.0 for proper probability distribution
        total_score = sum(combined.values())

        if (total_score > 0):
            combined = {model: score / total_score for model, score in combined.items()}
        
        # Calculate metric contributions for explainability
        if metric_scores:
            total_metric_impact = sum(metric_scores.values())
            if (total_metric_impact > 0):
                for model, score in metric_scores.items():
                    metric_contributions[model.value] = score / total_metric_impact
        
        return combined, metric_contributions


    def _make_domain_aware_prediction(self, combined_scores: Dict[str, float], domain: Domain, domain_preferences: List[AIModel]) -> Tuple[AIModel, float]:
        """
        Domain aware prediction that considers domain-specific model preferences - FIXED
        """
        if not combined_scores:
            return AIModel.UNKNOWN, 0.0
        
        # Find the model with the highest probability
        sorted_models = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_models:
            return AIModel.UNKNOWN, 0.0
        
        best_model_name, best_score = sorted_models[0]
        
        # FIXED: Only return UNKNOWN if the best score is very low
        # Use a more reasonable threshold for attribution
        if best_score < 0.05:  # Changed from 0.08 to 0.05 to be less restrictive
            return AIModel.UNKNOWN, best_score
        
        # FIXED: Don't override with domain preferences if there's a clear winner
        # Only consider domain preferences if scores are very close
        if len(sorted_models) > 1:
            second_model_name, second_score = sorted_models[1]
            score_difference = best_score - second_score
            
            # If scores are very close (within 3%) and second is domain-preferred, consider it
            if score_difference < 0.03:
                try:
                    best_model = AIModel(best_model_name)
                    second_model = AIModel(second_model_name)
                    
                    # If second model is domain-preferred and first is not, prefer second
                    if (second_model in domain_preferences and 
                        best_model not in domain_preferences):
                        best_model_name = second_model_name
                        best_score = second_score
                except ValueError:
                    pass
        
        try:
            best_model = AIModel(best_model_name)
        except ValueError:
            best_model = AIModel.UNKNOWN
        
        # Calculate confidence based on score dominance
        if len(sorted_models) > 1:
            second_score = sorted_models[1][1]
            margin = best_score - second_score
            # Confidence based on both absolute score and margin
            confidence = min(1.0, best_score * 0.6 + margin * 2.0)
        else:
            confidence = best_score * 0.7
        
        # FIXED: Don't downgrade to UNKNOWN based on confidence alone
        # If we have a model with reasonable probability, show it even with low confidence
        return best_model, confidence


    def _generate_detailed_reasoning(self, predicted_model: AIModel, confidence: float, domain: Domain, metric_contributions: Dict[str, float], 
                                     combined_scores: Dict[str, float]) -> List[str]:
        """
        Generate Explainable reasoning - FIXED to show proper formatting
        """
        reasoning = []
        
        reasoning.append("**AI Model Attribution Analysis**")
        reasoning.append("")
        reasoning.append(f"**Domain**: {domain.value.replace('_', ' ').title()}")
        reasoning.append("")
        
        # Show prediction with confidence
        if (predicted_model == AIModel.UNKNOWN):
            reasoning.append("**Most Likely**: Unable to determine with high confidence")
            reasoning.append("")
            reasoning.append("**Top Candidates:**")
        
        else:
            model_name = predicted_model.value.replace("-", " ").replace("_", " ").title()
            reasoning.append(f"**Predicted Model**: {model_name}")
            reasoning.append(f"**Confidence**: {confidence*100:.1f}%")
            reasoning.append("")
            reasoning.append("**Model Probability Distribution:**")
        
        reasoning.append("")
        
        # Show top candidates in proper format
        if combined_scores:
            sorted_models = sorted(combined_scores.items(), key = lambda x: x[1], reverse = True)
            
            for i, (model_name, score) in enumerate(sorted_models[:6]):
                # Skip very low probability models
                if (score < 0.01):  
                    continue
                
                display_name = model_name.replace("-", " ").replace("_", " ").title()
                percentage   = score * 100
                
                # Single line format: "• Model Name: XX.X%"
                reasoning.append(f"• **{display_name}**: {percentage:.1f}%")
        
        reasoning.append("")
        
        # Domain-specific insights - FIXED: Removed duplicate header
        reasoning.append("**Analysis Notes:**")
        reasoning.append(f"• Calibrated for {domain.value.replace('_', ' ')} domain")
        
        if (domain in [Domain.ACADEMIC, Domain.TECHNICAL_DOC, Domain.AI_ML, Domain.SOFTWARE_DEV, Domain.ENGINEERING, Domain.SCIENCE]):
            reasoning.append("• Higher weight on structural coherence and technical patterns")
        
        elif (domain in [Domain.CREATIVE, Domain.MARKETING, Domain.SOCIAL_MEDIA, Domain.BLOG_PERSONAL]):
            reasoning.append("• Emphasis on linguistic diversity and stylistic variation")
        
        elif (domain in [Domain.LEGAL, Domain.MEDICAL]):
            reasoning.append("• Focus on formal language and specialized terminology")
        
        elif (domain in [Domain.BUSINESS, Domain.JOURNALISM, Domain.TUTORIAL]):
            reasoning.append("• Balanced analysis across multiple attribution factors")
        
        return reasoning


    def _get_top_fingerprints(self, fingerprint_scores: Dict[AIModel, float]) -> Dict[str, int]:
        """
        Get top fingerprint matches for display
        """
        top_matches   = dict()
        sorted_models = sorted(fingerprint_scores.items(), key = lambda x: x[1], reverse = True)[:5]
        
        for model, score in sorted_models:
            # Only show meaningful matches
            if (score > 0.1):  
                top_matches[model.value] = int(score * 100)
        
        return top_matches


    def _create_unknown_result(self, domain: Domain) -> AttributionResult:
        """
        Create result for unknown attribution with domain context
        """
        return AttributionResult(predicted_model      = AIModel.UNKNOWN,
                                 confidence           = 0.0,
                                 model_probabilities  = {},
                                 reasoning            = [f"Model attribution inconclusive for {domain.value} content. Text may be human-written or from unidentifiable model"],
                                 fingerprint_matches  = {},
                                 domain_used          = domain,
                                 metric_contributions = {},
                                )


# Export
__all__ = ["AIModel", 
           "ModelAttributor", 
           "AttributionResult",
          ]