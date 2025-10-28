# DEPENDENCIES
from abc import ABC
from enum import Enum
from typing import Any
from typing import Dict
from typing import Tuple
from loguru import logger
from typing import Optional
from abc import abstractmethod
from dataclasses import dataclass


class MetricResult:
    """
    Result from a metric calculation
    """
    def __init__(self, metric_name: str, ai_probability: float, human_probability: float, mixed_probability: float, confidence: float, details: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.metric_name       = metric_name
        self.ai_probability    = max(0.0, min(1.0, ai_probability))
        self.human_probability = max(0.0, min(1.0, human_probability))
        self.mixed_probability = max(0.0, min(1.0, mixed_probability))
        self.confidence        = max(0.0, min(1.0, confidence))
        self.details           = details or {}
        self.error             = error
        
        # Normalize probabilities to sum to 1
        total                  = self.ai_probability + self.human_probability + self.mixed_probability
        
        if (total > 0):
            self.ai_probability    /= total
            self.human_probability /= total
            self.mixed_probability /= total

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"metric_name"       : self.metric_name,
                "ai_probability"    : round(self.ai_probability, 4),
                "human_probability" : round(self.human_probability, 4),
                "mixed_probability" : round(self.mixed_probability, 4),
                "confidence"        : round(self.confidence, 4),
                "details"           : self.details,
                "error"             : self.error,
                "success"           : self.error is None,
               }
    

    @property
    def is_ai(self) -> bool:
        """
        Check if classified as AI
        """
        return self.ai_probability > max(self.human_probability, self.mixed_probability)
    
    
    @property
    def is_human(self) -> bool:
        """
        Check if classified as human
        """
        return self.human_probability > max(self.ai_probability, self.mixed_probability)
    

    @property
    def is_mixed(self) -> bool:
        """
        Check if classified as mixed
        """
        return self.mixed_probability > max(self.ai_probability, self.human_probability)
    

    @property
    def predicted_class(self) -> str:
        """
        Get predicted class
        """
        if self.is_ai:
            return "AI"
        
        elif self.is_human:
            return "Human"
        
        else:
            return "Mixed"


class BaseMetric(ABC):
    """
    Abstract base class for all detection metrics
    """
    def __init__(self, name: str, description: str):
        self.name           = name
        self.description    = description
        self.is_initialized = False
        self._model         = None
        self._tokenizer     = None
    

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the metric (load models, etc.)
        
        Returns:
        --------
            True if successful, False otherwise
        """
        pass
    

    @abstractmethod
    def compute(self, text: str, **kwargs) -> MetricResult:
        """
        Compute the metric for given text
        
        Arguments:
        ----------
            text     { str } : Input text to analyze

            **kwargs         : Additional parameters
            
        Returns:
        --------
            MetricResult object
        """
        pass
    

    def cleanup(self):
        """
        Clean up resources
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self.is_initialized = False
    

    def __enter__(self):
        """
        Context manager entry
        """
        if not self.is_initialized:
            self.initialize()
        
        return self
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit
        """
        self.cleanup()

    
    def _safe_compute(self, text: str, **kwargs) -> MetricResult:
        """
        Safe wrapper for compute with error handling
        
        Arguments:
        ----------
            text     { str } : Input text

            **kwargs         : Additional parameters
            
        Returns:
        --------
            { MetricResult } : MetricResult (with error if computation failed)
        """
        try:
            if not self.is_initialized:
                logger.warning(f"{self.name}: Not initialized, initializing now...")
                if not self.initialize():
                    return MetricResult(metric_name       = self.name,
                                        ai_probability    = 0.5,
                                        human_probability = 0.5,
                                        mixed_probability = 0.0,
                                        confidence        = 0.0,
                                        error             = "Failed to initialize metric",
                                       )
            
            result = self.compute(text, **kwargs)
            return result
            

        except Exception as e:
            logger.error(f"{self.name}: Error computing metric: {e}")
            return MetricResult(metric_name       = self.name,
                                ai_probability    = 0.5,
                                human_probability = 0.5,
                                mixed_probability = 0.0,
                                confidence        = 0.0,
                                error             = str(e),
                               )

    
    def batch_compute(self, texts: list, **kwargs) -> list:
        """
        Compute metric for multiple texts
        
        Arguments:
        ----------
            texts    { list } : List of input texts

            **kwargs          : Additional parameters
            
        Returns:
        --------
               { list }       : List of MetricResult objects
        """
        results = list()

        for text in texts:
            result = self._safe_compute(text, **kwargs)
            results.append(result)
        
        return results
    

    def get_info(self) -> Dict[str, Any]:
        """
        Get metric information
        """
        return {"name"        : self.name,
                "description" : self.description,
                "initialized" : self.is_initialized,
               }
    

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.is_initialized})"



class StatisticalMetric(BaseMetric):
    """
    Base class for statistical metrics that don't require models
    """
    
    def initialize(self) -> bool:
        """
        Statistical metrics don't need initialization
        """
        self.is_initialized = True
        return True



# Export
__all__ = ["BaseMetric",
           "MetricResult",
           "StatisticalMetric",
          ]