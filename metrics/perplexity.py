# DEPENDENCIES
import re
import math
import torch
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from config.threshold_config import Domain
from metrics.base_metric import BaseMetric
from metrics.base_metric import MetricResult
from models.model_manager import get_model_manager
from config.threshold_config import get_threshold_for_domain


class PerplexityMetric(BaseMetric):
    """
    Text predictability analysis using GPT-2 for perplexity calculation
    
    Measures (Aligned with Documentation):
    - Overall text perplexity (lower = more predictable = more AI-like)
    - Perplexity distribution across text chunks
    - Sentence-level perplexity patterns
    - Cross-entropy analysis
    """
    def __init__(self):
        super().__init__(name        = "perplexity",
                         description = "GPT-2 based perplexity calculation for text predictability analysis",
                        )

        self.model     = None
        self.tokenizer = None
    

    def initialize(self) -> bool:
        """
        Initialize the perplexity metric
        """
        try:
            logger.info("Initializing perplexity metric...")
            
            # Load GPT-2 model and tokenizer
            model_manager = get_model_manager()
            model_result  = model_manager.load_model(model_name = "perplexity_gpt2")
            
            if isinstance(model_result, tuple):
                self.model, self.tokenizer = model_result
            
            else:
                logger.error("Failed to load GPT-2 model for perplexity calculation")
                return False
            
            self.is_initialized = True
            logger.success("Perplexity metric initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize perplexity metric: {repr(e)}")
            return False
    

    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute perplexity measures with FULL DOMAIN THRESHOLD INTEGRATION
        """
        try:
            if not text or len(text.strip()) < 50:
                return MetricResult(metric_name       = self.name,
                                    ai_probability    = 0.5,
                                    human_probability = 0.5,
                                    mixed_probability = 0.0,
                                    confidence        = 0.1,
                                    error             = "Text too short for perplexity analysis",
                                   )
            
            # Get domain-specific thresholds
            domain                           = kwargs.get('domain', Domain.GENERAL)
            domain_thresholds                = get_threshold_for_domain(domain)
            perplexity_thresholds            = domain_thresholds.perplexity
            
            # Calculate comprehensive perplexity features
            features                         = self._calculate_perplexity_features(text)
            
            # Calculate raw perplexity score (0-1 scale)
            raw_perplexity_score, confidence = self._analyze_perplexity_patterns(features)
            
            # Apply domain-specific thresholds to convert raw score to probabilities
            ai_prob, human_prob, mixed_prob  = self._apply_domain_thresholds(raw_perplexity_score, perplexity_thresholds, features)
            
            # Apply confidence multiplier from domain thresholds
            confidence                      *= perplexity_thresholds.confidence_multiplier
            confidence                       = max(0.0, min(1.0, confidence))
            
            return MetricResult(metric_name       = self.name,
                                ai_probability    = ai_prob,
                                human_probability = human_prob,
                                mixed_probability = mixed_prob,
                                confidence        = confidence,
                                details           = {**features, 
                                                     'domain_used'     : domain.value,
                                                     'ai_threshold'    : perplexity_thresholds.ai_threshold,
                                                     'human_threshold' : perplexity_thresholds.human_threshold,
                                                     'raw_score'       : raw_perplexity_score,
                                                    },
                               )
            
        except Exception as e:
            logger.error(f"Error in perplexity computation: {repr(e)}")
            return MetricResult(metric_name       = self.name,
                                ai_probability    = 0.5,
                                human_probability = 0.5,
                                mixed_probability = 0.0,
                                confidence        = 0.0,
                                error             = str(e),
                               )
    

    def _apply_domain_thresholds(self, raw_score: float, thresholds: Any, features: Dict[str, Any]) -> tuple:
        """
        Apply domain-specific thresholds to convert raw score to probabilities
        """
        ai_threshold    = thresholds.ai_threshold      # e.g., 0.60 for GENERAL, 0.55 for ACADEMIC
        human_threshold = thresholds.human_threshold   # e.g., 0.40 for GENERAL, 0.35 for ACADEMIC
        
        # Calculate probabilities based on threshold distances
        if (raw_score >= ai_threshold):
            # Above AI threshold - strongly AI
            distance_from_threshold = raw_score - ai_threshold
            ai_prob                 = 0.7 + (distance_from_threshold * 0.3)  # 0.7 to 1.0
            human_prob              = 0.3 - (distance_from_threshold * 0.3)  # 0.3 to 0.0

        elif (raw_score <= human_threshold):
            # Below human threshold - strongly human
            distance_from_threshold = human_threshold - raw_score
            ai_prob                 = 0.3 - (distance_from_threshold * 0.3)  # 0.3 to 0.0
            human_prob              = 0.7 + (distance_from_threshold * 0.3)  # 0.7 to 1.0
        
        else:
            # Between thresholds - uncertain zone
            range_width = ai_threshold - human_threshold

            if (range_width > 0):
                position_in_range = (raw_score - human_threshold) / range_width
                ai_prob           = 0.3 + (position_in_range * 0.4)  # 0.3 to 0.7
                human_prob        = 0.7 - (position_in_range * 0.4)  # 0.7 to 0.3
            
            else:
                ai_prob = 0.5
                human_prob = 0.5
        
        # Ensure probabilities are valid
        ai_prob    = max(0.0, min(1.0, ai_prob))
        human_prob = max(0.0, min(1.0, human_prob))
        
        # Calculate mixed probability based on perplexity variance
        mixed_prob = self._calculate_mixed_probability(features)
        
        # Normalize to sum to 1.0
        total      = ai_prob + human_prob + mixed_prob

        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return ai_prob, human_prob, mixed_prob
    

    def _calculate_perplexity_features(self, text: str) -> Dict[str, Any]:
        """
        Calculate comprehensive perplexity measures
        """
        if not self.model or not self.tokenizer:
            return self._get_default_features()
        
        # Calculate overall perplexity
        overall_perplexity    = self._calculate_perplexity(text)
        
        # Split into sentences for sentence-level analysis
        sentences             = self._split_sentences(text)
        
        # Calculate sentence-level perplexities
        sentence_perplexities = list()
        valid_sentences       = 0
        
        for sentence in sentences:
            # Minimum sentence length
            if (len(sentence.strip()) > 20):  
                sent_perplexity = self._calculate_perplexity(sentence)
                
                if (sent_perplexity > 0):
                    sentence_perplexities.append(sent_perplexity)
                    valid_sentences += 1
        
        # Calculate statistical features
        if sentence_perplexities:
            avg_sentence_perplexity = np.mean(sentence_perplexities)
            std_sentence_perplexity = np.std(sentence_perplexities)
            min_sentence_perplexity = np.min(sentence_perplexities)
            max_sentence_perplexity = np.max(sentence_perplexities)
        
        else:
            avg_sentence_perplexity = overall_perplexity
            std_sentence_perplexity = 0.0
            min_sentence_perplexity = overall_perplexity
            max_sentence_perplexity = overall_perplexity
        
        # Chunk-based analysis for whole-text understanding
        chunk_perplexities    = self._calculate_chunk_perplexity(text, chunk_size = 200)
        perplexity_variance   = np.var(chunk_perplexities) if chunk_perplexities else 0.0
        avg_chunk_perplexity  = np.mean(chunk_perplexities) if chunk_perplexities else overall_perplexity
        
        # Normalize perplexity to 0-1 scale for easier interpretation
        normalized_perplexity = self._normalize_perplexity(overall_perplexity)
        
        # Cross-entropy analysis
        cross_entropy_score   = self._calculate_cross_entropy(text)
        
        return {"overall_perplexity"      : round(overall_perplexity, 2),
                "normalized_perplexity"   : round(normalized_perplexity, 4),
                "avg_sentence_perplexity" : round(avg_sentence_perplexity, 2),
                "std_sentence_perplexity" : round(std_sentence_perplexity, 2),
                "min_sentence_perplexity" : round(min_sentence_perplexity, 2),
                "max_sentence_perplexity" : round(max_sentence_perplexity, 2),
                "perplexity_variance"     : round(perplexity_variance, 4),
                "avg_chunk_perplexity"    : round(avg_chunk_perplexity, 2),
                "cross_entropy_score"     : round(cross_entropy_score, 4),
                "num_sentences_analyzed"  : valid_sentences,
                "num_chunks_analyzed"     : len(chunk_perplexities),
               }
    

    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity for given text using GPT-2 : Lower perplexity = more predictable = more AI-like
        """
        try:
            # Check text length before tokenization
            if (len(text.strip()) < 10):
                return 0.0

            # Tokenize the text
            encodings = self.tokenizer(text, 
                                       return_tensors = 'pt', 
                                       truncation     = True, 
                                       max_length     = 1024,
                                      )

            input_ids = encodings.input_ids
            
            # Minimum tokens
            if ((input_ids.numel() == 0) or (input_ids.size(1) < 5)):
                return 0.0
            
            # Calculate loss (cross-entropy)
            with torch.no_grad():
                outputs = self.model(input_ids, labels = input_ids)
                loss    = outputs.loss
            
            # Convert loss to perplexity
            perplexity = math.exp(loss.item())

            return perplexity
            
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {repr(e)}")
            return 0.0
    

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    

    def _calculate_chunk_perplexity(self, text: str, chunk_size: int = 200) -> List[float]:
        """
        Calculate perplexity across text chunks for whole-text analysis
        """
        chunks = list()
        words  = text.split()

        # Ensure we have enough words for meaningful chunks
        if (len(words) < chunk_size // 2):
            return [self._calculate_perplexity(text)] if text.strip() else []
        
        # Create overlapping chunks for better analysis
        for i in range(0, len(words), chunk_size // 2):
            chunk = ' '.join(words[i:i + chunk_size])
            
            # Minimum chunk size
            if (len(chunk) > 50):  
                perplexity = self._calculate_perplexity(chunk)

                # Reasonable range check
                if ((perplexity > 0) and (perplexity < 1000)):
                    chunks.append(perplexity)
        
        return chunks if chunks else [0.0]
    

    def _normalize_perplexity(self, perplexity: float) -> float:
        """
        Normalize perplexity using sigmoid transformation

        Lower perplexity = higher normalized score = more AI-like
        """
        # Use exponential normalization : Typical ranges: AI = 10-40, Human = 20-100
        normalized = 1.0 / (1.0 + np.exp((perplexity - 30) / 10))
        
        return normalized 
    

    def _calculate_cross_entropy(self, text: str) -> float:
        """
        Calculate cross-entropy as an alternative measure
        """
        try:
            encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = encodings.input_ids
            
            if (input_ids.numel() == 0):
                return 0.0
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels = input_ids)
                loss    = outputs.loss
            
            # Normalize cross-entropy to 0-1 scale : Assuming max ~5 nats
            cross_entropy = loss.item()
            normalized_ce = min(1.0, cross_entropy / 5.0) 
            
            return normalized_ce
            
        except Exception as e:
            logger.warning(f"Cross-entropy calculation failed: {repr(e)}")
            return 0.0
    

    def _analyze_perplexity_patterns(self, features: Dict[str, Any]) -> tuple:
        """
        Analyze perplexity patterns to determine RAW perplexity score (0-1 scale) : Higher score = more AI-like
        """
        # Check feature validity first
        required_features = ['normalized_perplexity', 'perplexity_variance', 'std_sentence_perplexity', 'cross_entropy_score']
    
        valid_features    = [features.get(feat, 0) for feat in required_features if features.get(feat, 0) > 0]
    
        if (len(valid_features) < 3):
            # Low confidence if insufficient features
            return 0.5, 0.3  


        # Initialize ai_indicator list
        ai_indicators = list()
        
        # Low overall perplexity suggests AI
        if (features['normalized_perplexity'] > 0.7):
            # Very AI-like
            ai_indicators.append(0.8)  

        elif (features['normalized_perplexity'] > 0.5):
            # AI-like
            ai_indicators.append(0.6)  

        else:
            # Human-like
            ai_indicators.append(0.2) 
        
        # Low perplexity variance suggests AI (consistent predictability)
        if (features['perplexity_variance'] < 50):
            ai_indicators.append(0.7)

        elif (features['perplexity_variance'] < 200):
            ai_indicators.append(0.4)

        else:
            ai_indicators.append(0.2)
        
        # Low sentence perplexity std suggests AI (consistent across sentences)
        if (features['std_sentence_perplexity'] < 20):
            ai_indicators.append(0.8)

        elif (features['std_sentence_perplexity'] < 50):
            ai_indicators.append(0.5)
        
        else:
            ai_indicators.append(0.2)
        
        # Low cross-entropy suggests AI (more predictable)
        if (features['cross_entropy_score'] < 0.3):
            ai_indicators.append(0.7)
        
        elif (features['cross_entropy_score'] < 0.6):
            ai_indicators.append(0.4)
        
        else:
            ai_indicators.append(0.2)
        
        # Consistent chunk perplexity suggests AI
        chunk_variance = features['perplexity_variance']
        
        if (chunk_variance < 25):
            ai_indicators.append(0.9)

        elif (chunk_variance < 100):
            ai_indicators.append(0.6)

        else:
            ai_indicators.append(0.3)
        
        # Calculate raw score and confidence
        raw_score  = np.mean(ai_indicators) if ai_indicators else 0.5
        confidence = 1.0 - (np.std(ai_indicators) / 0.5) if ai_indicators else 0.5
        confidence = max(0.1, min(0.9, confidence))
        
        return raw_score, confidence
    

    def _calculate_mixed_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of mixed AI/Human content
        """
        mixed_indicators = list()
        
        # Moderate perplexity values might indicate mixing
        if (0.4 <= features['normalized_perplexity'] <= 0.6):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        # High perplexity variance suggests mixed content
        if (features['perplexity_variance'] > 200):
            mixed_indicators.append(0.4)

        elif (features['perplexity_variance'] > 100):
            mixed_indicators.append(0.2)

        else:
            mixed_indicators.append(0.0)
        
        # Inconsistent sentence perplexities
        if (20 <= features['std_sentence_perplexity'] <= 60):
            mixed_indicators.append(0.3)

        else:
            mixed_indicators.append(0.0)
        
        return min(0.3, np.mean(mixed_indicators)) if mixed_indicators else 0.0
    

    def _get_default_features(self) -> Dict[str, Any]:
        """
        Return default features when analysis is not possible
        """
        return {"overall_perplexity"      : 50.0,
                "normalized_perplexity"   : 0.5,
                "avg_sentence_perplexity" : 50.0,
                "std_sentence_perplexity" : 25.0,
                "min_sentence_perplexity" : 30.0,
                "max_sentence_perplexity" : 70.0,
                "perplexity_variance"     : 100.0,
                "avg_chunk_perplexity"    : 50.0,
                "cross_entropy_score"     : 0.5,
                "num_sentences_analyzed"  : 0,
                "num_chunks_analyzed"     : 0,
               }
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.model     = None
        self.tokenizer = None
        super().cleanup()



# Export
__all__ = ["PerplexityMetric"]
