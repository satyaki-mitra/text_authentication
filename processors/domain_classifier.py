# DEPENDENCIES
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from typing import Optional
from dataclasses import dataclass
from config.threshold_config import Domain
from models.model_manager import get_model_manager
from config.threshold_config import interpolate_thresholds
from config.threshold_config import get_threshold_for_domain


@dataclass
class DomainPrediction:
    """
    Result of domain classification
    """
    primary_domain   : Domain
    secondary_domain : Optional[Domain]
    confidence       : float
    domain_scores    : Dict[str, float]
    

class DomainClassifier:
    """
    Classifies text into domains using zero-shot classification
    """
    # Enhanced domain labels for zero-shot classification
    DOMAIN_LABELS = {Domain.ACADEMIC      : ["academic paper", "research article", "scientific paper", "scholarly writing", "thesis", "dissertation", "academic research"],
                     Domain.CREATIVE      : ["creative writing", "fiction", "story", "narrative", "poetry", "literary work", "imaginative writing"],
                     Domain.AI_ML         : ["artificial intelligence", "machine learning", "neural networks", "data science", "AI research", "deep learning"],
                     Domain.SOFTWARE_DEV  : ["software development", "programming", "coding", "software engineering", "web development", "application development"],
                     Domain.TECHNICAL_DOC : ["technical documentation", "user manual", "API documentation", "technical guide", "system documentation"],
                     Domain.ENGINEERING   : ["engineering document", "technical design", "engineering analysis", "mechanical engineering", "electrical engineering"],
                     Domain.SCIENCE       : ["scientific research", "physics", "chemistry", "biology", "scientific study", "experimental results"],
                     Domain.BUSINESS      : ["business document", "corporate communication", "business report", "professional writing", "executive summary"],
                     Domain.JOURNALISM    : ["news article", "journalism", "press release", "news report", "media content", "reporting"],
                     Domain.SOCIAL_MEDIA  : ["social media post", "casual writing", "online content", "informal text", "social media content"],
                     Domain.BLOG_PERSONAL : ["personal blog", "personal writing", "lifestyle blog", "personal experience", "opinion piece", "diary entry"],
                     Domain.LEGAL         : ["legal document", "contract", "legal writing", "law", "legal agreement", "legal analysis"],
                     Domain.MEDICAL       : ["medical document", "healthcare", "clinical", "medical report", "health information", "medical research"],
                     Domain.MARKETING     : ["marketing content", "advertising", "brand content", "promotional writing", "sales copy", "marketing material"],
                     Domain.TUTORIAL      : ["tutorial", "how-to guide", "instructional content", "step-by-step guide", "educational guide", "learning material"],
                     Domain.GENERAL       : ["general content", "everyday writing", "common text", "standard writing", "normal text", "general information"],
                    }
    

    def __init__(self):
        self.model_manager       = get_model_manager()
        self.primary_classifier  = None
        self.fallback_classifier = None
        self.is_initialized      = False

    
    def initialize(self) -> bool:
        """
        Initialize the domain classifier with zero-shot models
        """
        try:
            logger.info("Initializing domain classifier...")
            
            # Load primary domain classifier (zero-shot)
            self.primary_classifier = self.model_manager.load_model(model_name = "domain_classifier")
            
            # Load fallback classifier
            try:
                self.fallback_classifier = self.model_manager.load_model(model_name = "domain_classifier_fallback")
                logger.info("Fallback classifier loaded successfully")

            except Exception as e:
                logger.warning(f"Could not load fallback classifier: {repr(e)}")
                self.fallback_classifier = None
            
            self.is_initialized = True
            logger.success("Domain classifier initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize domain classifier: {repr(e)}")
            return False
    

    def classify(self, text: str, top_k: int = 2, min_confidence: float = 0.3) -> DomainPrediction:
        """
        Classify text into domain using zero-shot classification
        
        Arguments:
        ----------
            text            { str }  : Input text

            top_k           { int }  : Number of top domains to consider
            
            min_confidence { float } : Minimum confidence threshold
            
        Returns:
        --------
            { DomainPrediction }     : DomainPrediction object
        """
        if not self.is_initialized:
            logger.warning("Domain classifier not initialized, initializing now...")
            if not self.initialize():
                return self._get_default_prediction()
        
        try:
            # First try with primary classifier
            primary_result = self._classify_with_model(text       = text, 
                                                       classifier = self.primary_classifier, 
                                                       model_type = "primary",
                                                      )
            
            # If primary result meets confidence threshold, return it
            if (primary_result.confidence >= min_confidence):
                return primary_result
            
            # If primary is low confidence but we have fallback, try fallback
            if self.fallback_classifier:
                logger.info("Primary classifier low confidence, trying fallback model...")
                fallback_result = self._classify_with_model(text       = text, 
                                                            classifier = self.fallback_classifier, 
                                                            model_type = "fallback",
                                                           )
                
                # Use fallback if it has higher confidence
                if fallback_result.confidence > primary_result.confidence:
                    return fallback_result
            
            # Return primary result even if low confidence
            return primary_result
            
        except Exception as e:
            logger.error(f"Error in domain classification: {repr(e)}")
            
            # Try fallback classifier if primary failed
            if self.fallback_classifier:
                try:
                    logger.info("Trying fallback classifier after primary failure...")
                    return self._classify_with_model(text       = text, 
                                                     classifier = self.fallback_classifier, 
                                                     model_type = "fallback",
                                                    )
                
                except Exception as fallback_error:
                    logger.error(f"Fallback classifier also failed: {repr(fallback_error)}")
            
            # Both models failed, return default
            return self._get_default_prediction()
    

    def _classify_with_model(self, text: str, classifier, model_type: str) -> DomainPrediction:
        """
        Classify using a zero-shot classification model
        """
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Get all candidate labels
        all_labels      = list()
        label_to_domain = dict()

        for domain, labels in self.DOMAIN_LABELS.items():
            # Use the first label as the primary one for this domain
            primary_label = labels[0]
            all_labels.append(primary_label)
            label_to_domain[primary_label] = domain
        
        # Perform zero-shot classification
        result = classifier(processed_text,
                            candidate_labels    = all_labels,
                            multi_label         = False,
                            hypothesis_template = "This text is about {}.",
                           )
        
        # Convert to domain scores
        domain_scores = dict()

        for label, score in zip(result['labels'], result['scores']):
            domain     = label_to_domain[label]
            domain_key = domain.value

            if (domain_key not in domain_scores):
                domain_scores[domain_key] = list()

            domain_scores[domain_key].append(score)
        
        # Average scores for each domain
        avg_domain_scores                 = {domain: sum(scores) / len(scores) for domain, scores in domain_scores.items()}
        
        # Sort by score
        sorted_domains                    = sorted(avg_domain_scores.items(), key = lambda x: x[1], reverse = True)
        
        # Get primary and secondary domains
        primary_domain_str, primary_score = sorted_domains[0]
        primary_domain                    = Domain(primary_domain_str)
        
        secondary_domain                  = None
        secondary_score                   = 0.0

        if ((len(sorted_domains) > 1) and (sorted_domains[1][1] >= 0.1)):
            secondary_domain = Domain(sorted_domains[1][0])
            secondary_score  = sorted_domains[1][1]
        
        # Calculate confidence
        confidence = primary_score
        
        # If we have mixed domains with close scores, adjust confidence
        if (secondary_domain and (primary_score < 0.7) and (secondary_score > 0.3)):
            score_ratio = secondary_score / primary_score
            
            # Secondary is at least 60% of primary
            if (score_ratio > 0.6):  
                # Lower confidence for mixed domains
                confidence = (primary_score + secondary_score) / 2 * 0.8
                logger.info(f"Mixed domain detected: {primary_domain.value} + {secondary_domain.value}, will use interpolated thresholds")
        
        # If primary score is low and we have a secondary, it's uncertain
        elif ((primary_score < 0.5) and secondary_domain):
            # Reduce confidence
            confidence *= 0.8  
        
        logger.info(f"{model_type.capitalize()} model classified domain: {primary_domain.value} (confidence: {confidence:.3f})")
        
        return DomainPrediction(primary_domain   = primary_domain,
                                secondary_domain = secondary_domain,
                                confidence       = confidence,
                                domain_scores    = avg_domain_scores,
                               )
    

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification
        """
        # Truncate to reasonable length
        words = text.split()
        if (len(words) > 400):
            text = ' '.join(words[:400])
        
        # Clean up text
        text = text.strip()
        if not text:
            return "general content"
        
        return text
    

    def _get_default_prediction(self) -> DomainPrediction:
        """
        Get default prediction when classification fails
        """
        return DomainPrediction(primary_domain   = Domain.GENERAL,
                                secondary_domain = None,
                                confidence       = 0.5,
                                domain_scores    = {Domain.GENERAL.value: 1.0},
                               )
    

    def get_adaptive_thresholds(self, domain_prediction: DomainPrediction):
        """
        Get adaptive thresholds based on domain prediction
        """
        if ((domain_prediction.confidence > 0.7) and (not domain_prediction.secondary_domain)):
            return get_threshold_for_domain(domain_prediction.primary_domain)
        
        if domain_prediction.secondary_domain:
            primary_score   = domain_prediction.domain_scores.get(domain_prediction.primary_domain.value, 0)
            secondary_score = domain_prediction.domain_scores.get(domain_prediction.secondary_domain.value, 0)
            
            if (primary_score + secondary_score > 0):
                weight1 = primary_score / (primary_score + secondary_score)
            
            else:
                weight1 = domain_prediction.confidence
                
            return interpolate_thresholds(domain1  = domain_prediction.primary_domain,
                                          domain2  = domain_prediction.secondary_domain,
                                          weight1  = weight1,
                                         )
        
        if (domain_prediction.confidence < 0.6):
            return interpolate_thresholds(domain1 = domain_prediction.primary_domain,
                                          domain2 = Domain.GENERAL,
                                          weight1 = domain_prediction.confidence,
                                         )
        
        return get_threshold_for_domain(domain_prediction.primary_domain)
    
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.primary_classifier  = None
        self.fallback_classifier = None
        self.is_initialized      = False



# Export
__all__ = ["DomainClassifier",
           "DomainPrediction",
          ]