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
    Classifies text into domains using primary model with different fallback model
    """
    # Domain labels for classification
    DOMAIN_LABELS = {Domain.ACADEMIC     : ["academic writing", "research paper", "scholarly article", "thesis", "scientific report"],
                     Domain.CREATIVE     : ["creative writing", "fiction", "poetry", "story", "narrative"],
                     Domain.AI_ML        : ["machine learning", "artificial intelligence", "neural networks", "data science", "AI research"],
                     Domain.SOFTWARE_DEV : ["software development", "programming", "coding", "software engineering", "web development"],
                     Domain.TECHNICAL_DOC: ["technical documentation", "user manual", "API documentation", "technical guide", "installation guide"],
                     Domain.ENGINEERING  : ["engineering", "mechanical engineering", "electrical engineering", "design", "technical design"],
                     Domain.SCIENCE      : ["scientific research", "physics", "chemistry", "biology", "scientific study"],
                     Domain.BUSINESS     : ["business document", "corporate communication", "professional writing", "business report", "marketing"],
                     Domain.JOURNALISM   : ["news article", "journalism", "press release", "news report", "media"],
                     Domain.SOCIAL_MEDIA : ["social media post", "blog post", "casual writing", "online content", "informal text"],
                     Domain.BLOG_PERSONAL: ["personal blog", "personal writing", "lifestyle blog", "personal experience", "opinion piece"],
                     Domain.LEGAL        : ["legal document", "contract", "legal writing", "law", "judicial"],
                     Domain.MEDICAL      : ["medical document", "healthcare", "clinical", "medical report", "health"],
                     Domain.MARKETING    : ["marketing content", "advertising", "brand content", "promotional writing", "sales copy"],
                     Domain.TUTORIAL     : ["tutorial", "how-to guide", "instructional content", "step-by-step guide", "educational guide"],
                     Domain.GENERAL      : ["general content", "everyday writing", "common text", "standard writing", "normal text"]
                    }
    

    def __init__(self):
        self.model_manager       = get_model_manager()
        self.primary_classifier  = None
        self.fallback_classifier = None
        self.is_initialized      = False

    
    def initialize(self) -> bool:
        """
        Initialize the domain classifier with primary and fallback models
        """
        try:
            logger.info("Initializing domain classifier...")
            
            # Load primary domain classifier (distilbert-based)
            self.primary_classifier = self.model_manager.load_pipeline(model_name = "domain_classifier",
                                                                       task       = "zero-shot-classification",
                                                                      )
            
            # Load fallback classifier (different model for robustness)
            try:
                self.fallback_classifier = self.model_manager.load_pipeline(model_name = "domain_classifier_fallback",
                                                                            task       = "zero-shot-classification",
                                                                           )
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
        Classify text into domain using primary model with fallback to different model
        
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
                if (fallback_result.confidence > primary_result.confidence):
                    return fallback_result
            
            # Return primary result even if low confidence
            return primary_result
            
        except Exception as e:
            logger.error(f"Error in primary domain classification: {repr(e)}")
            
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
        Classify using a specific model with interpolation for mixed domains
        
        Arguments:
        ----------
            text         { str }     : Input text

            classifier   { object }  : Classifier model
            
            model_type   { str }     : Type of model for logging
            
        Returns:
        --------
            { DomainPrediction }     : DomainPrediction object
        """
        # Truncate text if too long (keep first 500 words)
        words = text.split()
        if (len(words) > 500):
            text = ' '.join(words[:500])
        
        # Get all domain labels
        all_labels      = list()
        label_to_domain = dict()

        for domain, labels in self.DOMAIN_LABELS.items():
            for label in labels:
                all_labels.append(label)
                label_to_domain[label] = domain
        
        # Perform zero-shot classification
        result        = classifier(text,
                                   candidate_labels = all_labels,
                                   multi_label      = False,
                                  )
        
        # Aggregate scores by domain
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

        if ((len(sorted_domains) > 1) and (sorted_domains[1][1] >= 0.2)):  # Lower threshold for secondary
            secondary_domain = Domain(sorted_domains[1][0])
            secondary_score  = sorted_domains[1][1]
        
        # Calculate if we should use interpolated domain classification
        should_interpolate     = False
        interpolation_weight   = 0.5

        if (secondary_domain and (primary_score < 0.7) and (secondary_score > 0.3)):
            # If scores are close and both domains are significant, flag for interpolation
            score_ratio = secondary_score / primary_score
            
            # Secondary is at least 60% of primary
            if (score_ratio > 0.6):  
                should_interpolate   = True
                interpolation_weight = primary_score / (primary_score + secondary_score)
        
        # Calculate confidence
        confidence = primary_score
        
        # If we have mixed domains with interpolation, adjust confidence
        if (should_interpolate):
            # Lower confidence for mixed domains
            confidence = (primary_score + secondary_score) / 2 * 0.8
            logger.info(f"Mixed domain detected: {primary_domain.value} + {secondary_domain.value}, will use interpolated thresholds")
        
        # If primary score is low and we have a secondary, it's uncertain
        elif ((primary_score < 0.5) and secondary_domain):
            # Reduce confidence
            confidence *= 0.8  
        
        logger.info(f"{model_type.capitalize()} model classified domain: {primary_domain.value} (confidence: {confidence:.2f})")
        
        return DomainPrediction(primary_domain   = primary_domain,
                                secondary_domain = secondary_domain,
                                confidence       = confidence,
                                domain_scores    = avg_domain_scores,
                               )
    

    def _get_default_prediction(self) -> DomainPrediction:
        """
        Get default prediction when classification fails
        """
        return DomainPrediction(primary_domain   = Domain.GENERAL,
                                secondary_domain = None,
                                confidence       = 0.5,
                                domain_scores    = {domain.value: 1.0/len(Domain) for domain in Domain},
                               )
    

    def get_adaptive_thresholds(self, domain_prediction: DomainPrediction):
        """
        Get adaptive thresholds based on domain prediction with intelligent interpolation
        
        Arguments:
        ----------
            domain_prediction : Domain prediction result
            
        Returns:
        --------
            DomainThresholds object
        """
        # If we have a clear primary domain with high confidence
        if ((domain_prediction.confidence > 0.7) and (not domain_prediction.secondary_domain)):
            return get_threshold_for_domain(domain_prediction.primary_domain)
        
        # If we have primary and secondary domains, interpolate (ENHANCED LOGIC)
        if domain_prediction.secondary_domain:
            # Calculate interpolation weight based on score ratio
            primary_score   = domain_prediction.domain_scores.get(domain_prediction.primary_domain.value, 0)
            secondary_score = domain_prediction.domain_scores.get(domain_prediction.secondary_domain.value, 0)
            
            if (primary_score + secondary_score > 0):
                weight1 = primary_score / (primary_score + secondary_score)

            else:
                weight1 = domain_prediction.confidence
                
            thresholds = interpolate_thresholds(domain1 = domain_prediction.primary_domain,
                                                domain2 = domain_prediction.secondary_domain,
                                                weight1 = weight1,
                                               )
            return thresholds
        
        # If low confidence single domain, blend with general
        if (domain_prediction.confidence < 0.6):
            thresholds = interpolate_thresholds(domain1 = domain_prediction.primary_domain,
                                                domain2 = Domain.GENERAL,
                                                weight1 = domain_prediction.confidence,
                                               )
            return thresholds
        
        # Use primary domain with default thresholds
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