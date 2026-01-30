# DEPENDENCIES
import os
import torch
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from typing import Optional
from config.enums import Domain
import torch.nn.functional as F
from config.schemas import DomainPrediction
from models.model_manager import get_model_manager
from config.constants import domain_classification_params
from config.threshold_config import interpolate_thresholds
from config.threshold_config import get_threshold_for_domain

# Device-agnostic safety
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DomainClassifier:
    """
    Classifies text into domains using zero-shot classification
    """
    # Use constants from config - map string keys to Domain enum
    DOMAIN_LABELS = {Domain.ACADEMIC      : domain_classification_params.DOMAIN_LABELS["academic"],
                     Domain.CREATIVE      : domain_classification_params.DOMAIN_LABELS["creative"],
                     Domain.AI_ML         : domain_classification_params.DOMAIN_LABELS["ai_ml"],
                     Domain.SOFTWARE_DEV  : domain_classification_params.DOMAIN_LABELS["software_dev"],
                     Domain.TECHNICAL_DOC : domain_classification_params.DOMAIN_LABELS["technical_doc"],
                     Domain.ENGINEERING   : domain_classification_params.DOMAIN_LABELS["engineering"],
                     Domain.SCIENCE       : domain_classification_params.DOMAIN_LABELS["science"],
                     Domain.BUSINESS      : domain_classification_params.DOMAIN_LABELS["business"],
                     Domain.JOURNALISM    : domain_classification_params.DOMAIN_LABELS["journalism"],
                     Domain.SOCIAL_MEDIA  : domain_classification_params.DOMAIN_LABELS["social_media"],
                     Domain.BLOG_PERSONAL : domain_classification_params.DOMAIN_LABELS["blog_personal"],
                     Domain.LEGAL         : domain_classification_params.DOMAIN_LABELS["legal"],
                     Domain.MEDICAL       : domain_classification_params.DOMAIN_LABELS["medical"],
                     Domain.MARKETING     : domain_classification_params.DOMAIN_LABELS["marketing"],
                     Domain.TUTORIAL      : domain_classification_params.DOMAIN_LABELS["tutorial"],
                     Domain.GENERAL       : domain_classification_params.DOMAIN_LABELS["general"],
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
            self.primary_classifier = self.model_manager.load_model(model_name = "content_domain_classifier")
            
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
    

    def classify(self, text: str, top_k: int = domain_classification_params.TOP_K_DOMAINS, min_confidence: float = domain_classification_params.MIN_CONFIDENCE_THRESHOLD) -> DomainPrediction:
        """
        Classify text into domain using zero-shot classification
        
        Arguments:
        ----------
            text           { str }   : Input text

            top_k          { int }   : Number of top domains to consider
            
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
            
            # Save it as best result
            best_result    = primary_result
            
            # If primary is low confidence but we have fallback, try fallback
            if (self.fallback_classifier and (primary_result.evidence_strength < domain_classification_params.HIGH_CONFIDENCE_THRESHOLD)):
                logger.info("Primary classifier shows low confidence, trying fallback model...")
                fallback_result = self._classify_with_model(text       = text, 
                                                            classifier = self.fallback_classifier, 
                                                            model_type = "fallback",
                                                           )
                
                # Use fallback if it has higher confidence
                if (fallback_result.evidence_strength > best_result.evidence_strength):
                    best_result = fallback_result

            # Hard Safety Gate: if not any solid evidence with great confidence to determine domain, fallback to General Domain
            if best_result.evidence_strength < domain_classification_params.ABS_DOMAIN_CONFIDENCE_THRESHOLD:
                logger.info(f"Domain confidence {best_result.evidence_strength:.3f} below hard threshold {domain_classification_params.ABS_DOMAIN_CONFIDENCE_THRESHOLD:.2f}; forcing GENERAL domain")
                return DomainPrediction(primary_domain    = Domain.GENERAL,
                                        secondary_domain  = None,
                                        evidence_strength = 0.5,
                                        domain_scores     = {Domain.GENERAL.value: 1.0},
                                       )
            
            # Return primary result even if low confidence
            return best_result
            
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
    

    # def _classify_with_model(self, text: str, classifier, model_type: str) -> DomainPrediction:
    #     """
    #     Classify using a zero-shot classification model
    #     """
    #     # Preprocess text
    #     processed_text  = self._preprocess_text(text)
        
    #     # Get all candidate labels
    #     all_labels      = list()
    #     label_to_domain = dict()

    #     for domain, labels in self.DOMAIN_LABELS.items():
    #         # Use the first label as the primary one for this domain
    #         primary_label                  = labels[0]
    #         all_labels.append(primary_label)

    #         label_to_domain[primary_label] = domain
        
    #     # Perform zero-shot classification
    #     result = classifier(processed_text,
    #                         candidate_labels    = all_labels,
    #                         multi_label         = False,
    #                         hypothesis_template = "This text is about {}.",
    #                        )
        
    #     # Convert to domain scores
    #     domain_scores = dict()

    #     for label, score in zip(result['labels'], result['scores']):
    #         domain     = label_to_domain[label]
    #         domain_key = domain.value

    #         if (domain_key not in domain_scores):
    #             domain_scores[domain_key] = list()

    #         domain_scores[domain_key].append(score)
        
    #     # Average scores for each domain
    #     avg_domain_scores                 = {domain: sum(scores) / len(scores) for domain, scores in domain_scores.items()}
        
    #     # Sort by score
    #     sorted_domains                    = sorted(avg_domain_scores.items(), key = lambda x: x[1], reverse = True)
        
    #     # Get primary and secondary domains
    #     primary_domain_str, primary_score = sorted_domains[0]
    #     primary_domain                    = Domain(primary_domain_str)
        
    #     secondary_domain                  = None
    #     secondary_score                   = 0.0
        
    #     # Use constant for secondary domain minimum score
    #     secondary_min_score               = domain_classification_params.SECONDARY_DOMAIN_MIN_SCORE

    #     if ((len(sorted_domains) > 1) and (sorted_domains[1][1] >= secondary_min_score)):
    #         secondary_domain = Domain(sorted_domains[1][0])
    #         secondary_score  = sorted_domains[1][1]
        
    #     # Calculate evidence_strength
    #     evidence_strength    = primary_score
        
    #     # Use constants for mixed domain detection
    #     high_conf_threshold  = domain_classification_params.HIGH_CONFIDENCE_THRESHOLD
    #     mixed_secondary_min  = domain_classification_params.MIXED_DOMAIN_SECONDARY_MIN
    #     mixed_ratio_thresh   = domain_classification_params.MIXED_DOMAIN_RATIO_THRESHOLD
    #     mixed_conf_penalty   = domain_classification_params.MIXED_DOMAIN_CONFIDENCE_PENALTY
        
    #     # If we have mixed domains with close scores, adjust confidence
    #     if (secondary_domain and (primary_score < high_conf_threshold) and (secondary_score > mixed_secondary_min)):
            
    #         score_ratio = secondary_score / primary_score
            
    #         # Secondary is at least 60% of primary
    #         if (score_ratio > mixed_ratio_thresh):
    #             # Lower confidence for mixed domains
    #             evidence_strength = ((primary_score + secondary_score) / 2 * mixed_conf_penalty)
    #             logger.info(f"Mixed domain detected: {primary_domain.value} + {secondary_domain.value}, will use interpolated thresholds")
        
    #     # Use constant for low confidence threshold
    #     low_conf_threshold = domain_classification_params.LOW_CONFIDENCE_THRESHOLD
        
    #     # If primary score is low and we have a secondary, it's uncertain
    #     if ((primary_score < low_conf_threshold) and secondary_domain):
    #         # Reduce confidence using penalty
    #         evidence_strength *= mixed_conf_penalty
        
    #     logger.info(f"{model_type.capitalize()} model classified domain: {primary_domain.value} (confidence: {evidence_strength:.3f})")
        
    #     return DomainPrediction(primary_domain    = primary_domain,
    #                             secondary_domain  = secondary_domain,
    #                             evidence_strength = evidence_strength,
    #                             domain_scores     = avg_domain_scores,
    #                            )

    def _classify_with_model(self, text: str, classifier, model_type: str) -> DomainPrediction:
        """
        Classify using a manual NLI-style zero-shot classifier (NO pipelines)
        """

        model, tokenizer = classifier

        processed_text   = self._preprocess_text(text)

        # Build labels
        all_labels      = list()
        label_to_domain = dict()

        for domain, labels in self.DOMAIN_LABELS.items():
            for label in labels[:3]: 
                all_labels.append(label)
                label_to_domain[label] = domain

        # NLI formulation
        premises   = [processed_text] * len(all_labels)
        hypotheses = [f"This text is a {label}." for label in all_labels]

        # Tokenize safely (device-agnostic)
        inputs     = tokenizer(premises,
                               hypotheses,
                               return_tensors = "pt",
                               padding        = True,
                               truncation     = True,
                               max_length     = 1024,  # HARD SAFETY CAP
                              )

        inputs     = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            logits = model(**inputs).logits

        # Resolve entailment index robustly (works for ALL devices/models)
        label2id       = {k.lower(): v for k, v in model.config.label2id.items()}

        # fallback: last column
        entailment_idx = (label2id.get("entailment") or label2id.get("entails") or (logits.shape[-1] - 1))

        probs          = F.softmax(logits, dim = -1)
        scores         = probs[:, entailment_idx].detach().cpu().tolist()

        # Aggregate per-domain scores
        domain_scores  = dict()

        for label, score in zip(all_labels, scores):
            domain     = label_to_domain[label]
            domain_key = domain.value
            domain_scores.setdefault(domain_key, []).append(score)

        # Average scores
        avg_domain_scores                 = {domain: sum(vals) / len(vals) for domain, vals in domain_scores.items()}

        # Sort
        sorted_domains                    = sorted(avg_domain_scores.items(),
                                                   key     = lambda x: x[1],
                                                   reverse = True,
                                                  )

        primary_domain_str, primary_score = sorted_domains[0]
        primary_domain                    = Domain(primary_domain_str)
        secondary_domain                  = None
        secondary_score                   = 0.0
        secondary_min_score               = domain_classification_params.SECONDARY_DOMAIN_MIN_SCORE

        if (len(sorted_domains) > 1) and (sorted_domains[1][1] >= secondary_min_score):
            secondary_domain = Domain(sorted_domains[1][0])
            secondary_score  = sorted_domains[1][1]

        evidence_strength   = primary_score

        # Mixed-domain confidence adjustment 
        high_conf_threshold = domain_classification_params.HIGH_CONFIDENCE_THRESHOLD
        mixed_secondary_min = domain_classification_params.MIXED_DOMAIN_SECONDARY_MIN
        mixed_ratio_thresh  = domain_classification_params.MIXED_DOMAIN_RATIO_THRESHOLD
        mixed_conf_penalty  = domain_classification_params.MIXED_DOMAIN_CONFIDENCE_PENALTY

        if secondary_domain and primary_score < high_conf_threshold and secondary_score > mixed_secondary_min:
            score_ratio = secondary_score / max(primary_score, 1e-6)
            
            if (score_ratio > mixed_ratio_thresh):
                evidence_strength = ((primary_score + secondary_score) / 2) * mixed_conf_penalty
                logger.info(f"Mixed domain detected: {primary_domain.value} + {secondary_domain.value}")

        low_conf_threshold = domain_classification_params.LOW_CONFIDENCE_THRESHOLD

        if ((primary_score < low_conf_threshold) and secondary_domain):
            evidence_strength *= mixed_conf_penalty

        logger.info(f"{model_type.capitalize()} model classified domain: {primary_domain.value} (confidence: {evidence_strength:.3f})")

        return DomainPrediction(primary_domain    = primary_domain,
                                secondary_domain  = secondary_domain,
                                evidence_strength = evidence_strength,
                                domain_scores     = avg_domain_scores,
                               )
    

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification
        """
        # Truncate to reasonable length using constant
        max_words = domain_classification_params.MAX_WORDS_FOR_CLASSIFICATION
        words     = text.split()
        
        if (len(words) > max_words):
            text = ' '.join(words[:max_words])
        
        # Clean up text
        text = text.strip()
        if not text:
            return "general content"
        
        return text
    

    def _get_default_prediction(self) -> DomainPrediction:
        """
        Get default prediction when classification fails
        """
        return DomainPrediction(primary_domain    = Domain.GENERAL,
                                secondary_domain  = None,
                                evidence_strength = 0.5,
                                domain_scores     = {Domain.GENERAL.value: 1.0},
                               )
    

    def get_adaptive_thresholds(self, domain_prediction: DomainPrediction):
        """
        Get adaptive thresholds based on domain prediction
        """
        # Use constants for threshold decisions
        high_conf_threshold = domain_classification_params.HIGH_CONFIDENCE_THRESHOLD
        med_conf_threshold  = domain_classification_params.MEDIUM_CONFIDENCE_THRESHOLD
        
        # High confidence, single domain - use domain-specific thresholds
        if ((domain_prediction.evidence_strength > high_conf_threshold) and (not domain_prediction.secondary_domain)):
            return get_threshold_for_domain(domain_prediction.primary_domain)
        
        # Mixed domains - interpolate between primary and secondary
        if domain_prediction.secondary_domain:
            primary_score   = domain_prediction.domain_scores.get(domain_prediction.primary_domain.value, 0)
            secondary_score = domain_prediction.domain_scores.get(domain_prediction.secondary_domain.value, 0)
            
            if (primary_score + secondary_score > 0):
                weight1 = primary_score / (primary_score + secondary_score)
            
            else:
                weight1 = domain_prediction.evidence_strength
                
            return interpolate_thresholds(domain1 = domain_prediction.primary_domain,
                                          domain2 = domain_prediction.secondary_domain,
                                          weight1 = weight1,
                                         )
        
        # Low/medium confidence - blend with general domain
        if (domain_prediction.evidence_strength < med_conf_threshold):
            return interpolate_thresholds(domain1 = domain_prediction.primary_domain,
                                          domain2 = Domain.GENERAL,
                                          weight1 = domain_prediction.evidence_strength,
                                         )
        
        # Default: use domain-specific thresholds
        return get_threshold_for_domain(domain_prediction.primary_domain)
    
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.primary_classifier  = None
        self.fallback_classifier = None
        self.is_initialized      = False


def quick_classify(text: str, **kwargs) -> DomainPrediction:
    """
    Quick domain classification with default settings
    
    Arguments:
    ----------
        text     { str } : Input text

        **kwargs         : Override settings
        
    Returns:
    --------
        { DomainPrediction } : DomainPrediction object
    """
    classifier = DomainClassifier()
    classifier.initialize()
    return classifier.classify(text, **kwargs)


def get_domain_name(domain: Domain) -> str:
    """
    Get human-readable domain name
    
    Arguments:
    ----------
        domain { Domain } : Domain enum value
        
    Returns:
    --------
        { str } : Human-readable domain name
    """
    domain_names = {Domain.ACADEMIC      : "Academic",
                    Domain.CREATIVE      : "Creative Writing",
                    Domain.AI_ML         : "AI/ML",
                    Domain.SOFTWARE_DEV  : "Software Development",
                    Domain.TECHNICAL_DOC : "Technical Documentation",
                    Domain.ENGINEERING   : "Engineering",
                    Domain.SCIENCE       : "Science",
                    Domain.BUSINESS      : "Business",
                    Domain.JOURNALISM    : "Journalism",
                    Domain.SOCIAL_MEDIA  : "Social Media",
                    Domain.BLOG_PERSONAL : "Personal Blog",
                    Domain.LEGAL         : "Legal",
                    Domain.MEDICAL       : "Medical",
                    Domain.MARKETING     : "Marketing",
                    Domain.TUTORIAL      : "Tutorial",
                    Domain.GENERAL       : "General",
                   }

    return domain_names.get(domain, "Unknown")


def is_technical_domain(domain: Domain) -> bool:
    """
    Check if domain is technical in nature
    
    Arguments:
    ----------
        domain { Domain } : Domain enum value
        
    Returns:
    --------
        { bool } : True if technical domain
    """
    technical_domains = {Domain.AI_ML,
                         Domain.SOFTWARE_DEV,
                         Domain.TECHNICAL_DOC,
                         Domain.ENGINEERING,
                         Domain.SCIENCE,
                        }

    return domain in technical_domains


def is_creative_domain(domain: Domain) -> bool:
    """
    Check if domain is creative in nature
    
    Arguments:
    ----------
        domain { Domain } : Domain enum value
        
    Returns:
    --------
        { bool } : True if creative domain
    """
    creative_domains = {Domain.CREATIVE,
                        Domain.JOURNALISM,
                        Domain.SOCIAL_MEDIA,
                        Domain.BLOG_PERSONAL,
                        Domain.MARKETING,
                       }

    return domain in creative_domains


def is_formal_domain(domain: Domain) -> bool:
    """
    Check if domain is formal in nature
    
    Arguments:
    ----------
        domain { Domain } : Domain enum value
        
    Returns:
    --------
        { bool } : True if formal domain
    """
    formal_domains = {Domain.ACADEMIC,
                      Domain.LEGAL,
                      Domain.MEDICAL,
                      Domain.BUSINESS,
                     }

    return domain in formal_domains


# Export
__all__ = ["Domain",
           "DomainClassifier",
           "DomainPrediction",
           "quick_classify",
           "get_domain_name",
           "is_technical_domain",
           "is_creative_domain",
           "is_formal_domain",
          ]
