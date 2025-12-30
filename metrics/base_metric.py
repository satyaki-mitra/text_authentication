# DEPENDENCIES
from abc import ABC
from typing import Any
from typing import Dict
from loguru import logger
from typing import Optional
from abc import abstractmethod
from config.schemas import MetricResult
from config.constants import base_metric_params


class BaseMetric(ABC):
    """
    Abstract base class for all detection metrics
    """
    def __init__(self, name: str, description: str):
        self.name                          = name
        self.description                   = description
        self.is_initialized                = False
        self._model                        = None
        self._tokenizer                    = None
        self.default_synthetic_probability = base_metric_params.DEFAULT_SYNTHETIC_PROBABILITY
        self.default_authentic_probability = base_metric_params.DEFAULT_AUTHENTIC_PROBABILITY
        self.default_hybrid_probability    = base_metric_params.DEFAULT_HYBRID_PROBABILITY
        self.default_confidence            = base_metric_params.DEFAULT_CONFIDENCE
    

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
                    return self._default_result(error = "Failed to initialize metric")
            
            result = self.compute(text, **kwargs)
            return result
            

        except Exception as e:
            logger.error(f"{self.name}: Error computing metric: {e}")
            return self._default_result(error = str(e))

    
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

    
    def _default_result(self, error: Optional[str] = None) -> MetricResult:
        """
        Default metric result for exception cases
        """
        return MetricResult(metric_name           = self.name,
                            synthetic_probability = self.default_synthetic_probability,
                            authentic_probability = self.default_authentic_probability,
                            hybrid_probability    = self.default_hybrid_probability,
                            confidence            = self.default_confidence,
                            error                 = error,
                           )



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
           "StatisticalMetric",
          ]