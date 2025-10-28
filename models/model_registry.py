# DEPENDENCIES
import gc
import torch
import threading
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from config.model_config import ModelConfig
from config.model_config import MODEL_REGISTRY
from config.model_config import get_model_config


@dataclass
class ModelUsageStats:
    """
    Lightweight model usage statistics
    """
    model_name               : str
    load_count               : int
    last_used                : datetime
    total_usage_time_seconds : float
    avg_usage_time_seconds   : float
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"model_name"               : self.model_name,
                "load_count"               : self.load_count,
                "last_used"                : self.last_used.isoformat(),
                "total_usage_time_seconds" : round(self.total_usage_time_seconds, 2),
                "avg_usage_time_seconds"   : round(self.avg_usage_time_seconds, 2),
               }


class ModelRegistry:
    """
    Model registry module for tracking model usage statistics and performance metrics
    
    Complements ModelManager by adding:
    - Usage analytics
    - Performance monitoring  
    - Model dependency tracking
    - Health checks (without duplicating ModelManager functionality)
    """
    def __init__(self):
        self.usage_stats          : Dict[str, ModelUsageStats]  = dict()
        self.dependency_graph     : Dict[str, List[str]]        = dict()
        self.performance_metrics  : Dict[str, Dict[str, float]] = dict()
        self.lock                                               = threading.RLock()
        
        # Initialize from MODEL_REGISTRY
        self._initialize_registry()
        
        logger.info("ModelRegistry initialized for usage tracking")
    

    def _initialize_registry(self):
        """
        Initialize registry with all known models
        """
        for model_name in MODEL_REGISTRY.keys():
            self.usage_stats[model_name] = ModelUsageStats(model_name               = model_name,
                                                           load_count               = 0,
                                                           last_used                = datetime.now(),
                                                           total_usage_time_seconds = 0.0,
                                                           avg_usage_time_seconds   = 0.0,
                                                          )
    

    def record_model_usage(self, model_name: str, usage_time_seconds: float = 0.0):
        """
        Record that a model was used
        
        Arguments:
        ----------
            model_name          { str }  : Name of the model used

            usage_time_seconds { float } : How long the model was used (if available)
        """
        with self.lock:
            if model_name not in self.usage_stats:
                # Auto-register unknown models
                self.usage_stats[model_name] = ModelUsageStats(model_name               = model_name,
                                                               load_count               = 0,
                                                               last_used                = datetime.now(),
                                                               total_usage_time_seconds = 0.0,
                                                               avg_usage_time_seconds   = 0.0,
                                                              )
            
            stats             = self.usage_stats[model_name]
            stats.load_count += 1
            stats.last_used   = datetime.now()
            
            if (usage_time_seconds > 0):
                stats.total_usage_time_seconds += usage_time_seconds
                stats.avg_usage_time_seconds    = stats.total_usage_time_seconds / stats.load_count
            
            logger.debug(f"Recorded usage for {model_name} (count: {stats.load_count})")
    

    def get_usage_stats(self, model_name: str) -> Optional[ModelUsageStats]:
        """
        Get usage statistics for a model
        """
        with self.lock:
            return self.usage_stats.get(model_name)
    

    def get_most_used_models(self, top_k: int = 5) -> List[ModelUsageStats]:
        """
        Get most frequently used models
        """
        with self.lock:
            sorted_models = sorted(self.usage_stats.values(), 
                                   key     = lambda x: x.load_count, 
                                   reverse = True,
                                  )

            return sorted_models[:top_k]
    

    def record_performance_metric(self, model_name: str, metric_name: str, value: float):
        """
        Record performance metrics for a model
        
        Arguments:
        ----------
            model_name    { str }  : Name of the model

            metric_name  { float } : Name of the metric (e.g., "inference_time_ms", "memory_peak_mb")
            
            value         { str }  : Metric value
        """
        with self.lock:
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = {}
            
            self.performance_metrics[model_name][metric_name] = value
    

    def get_performance_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a model
        """
        with self.lock:
            return self.performance_metrics.get(model_name, {})
    

    def add_dependency(self, model_name: str, depends_on: List[str]):
        """
        Add dependency information for a model
        
        Arguments:
        ----------
            model_name  { str }  : The model that has dependencies

            depends_on  { list } : List of model names this model depends on
        """
        with self.lock:
            self.dependency_graph[model_name] = depends_on
    

    def get_dependencies(self, model_name: str) -> List[str]:
        """
        Get dependencies for a model
        """
        with self.lock:
            return self.dependency_graph.get(model_name, [])
    

    def get_dependent_models(self, model_name: str) -> List[str]:
        """
        Get models that depend on the specified model
        """
        with self.lock:
            dependents = []
            
            for user_model, dependencies in self.dependency_graph.items():
                if model_name in dependencies:
                    dependents.append(user_model)
            
            return dependents
    

    def generate_usage_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive usage report
        """
        with self.lock:
            total_usage   = sum(stats.load_count for stats in self.usage_stats.values())
            active_models = [name for name, stats in self.usage_stats.items() if stats.load_count > 0]
            
            return {"timestamp"           : datetime.now().isoformat(),
                    "summary"             : {"total_models_tracked" : len(self.usage_stats),
                                             "active_models"        : len(active_models),
                                             "total_usage_count"    : total_usage,
                                            },
                    "most_used_models"    : [stats.to_dict() for stats in self.get_most_used_models(top_k = 10)],
                    "performance_metrics" : {model: metrics for model, metrics in self.performance_metrics.items()},
                    "dependency_graph"    : self.dependency_graph
                   }
    

    def reset_usage_stats(self, model_name: Optional[str] = None):
        """
        Reset usage statistics for a model or all models
        
        Arguments:
        ----------
            model_name { str } : Specific model to reset, or None for all models
        """
        with self.lock:
            if model_name:
                if model_name in self.usage_stats:
                    self.usage_stats[model_name] = ModelUsageStats(model_name               = model_name,
                                                                   load_count               = 0,
                                                                   last_used                = datetime.now(),
                                                                   total_usage_time_seconds = 0.0,
                                                                   avg_usage_time_seconds   = 0.0,
                                                                  )

                    logger.info(f"Reset usage stats for {model_name}")
            
            else:
                self._initialize_registry()
                logger.info("Reset usage stats for all models")
    

    def cleanup(self):
        """
        Clean up resources
        """
        with self.lock:
            self.usage_stats.clear()
            self.performance_metrics.clear()
            self.dependency_graph.clear()
            
            logger.info("ModelRegistry cleanup completed")


# Singleton instance
_model_registry_instance: Optional[ModelRegistry] = None
_registry_lock                                    = threading.Lock()


def get_model_registry() -> ModelRegistry:
    """
    Get singleton ModelRegistry instance
    """
    global _model_registry_instance
    
    if _model_registry_instance is None:
        with _registry_lock:
            if _model_registry_instance is None:
                _model_registry_instance = ModelRegistry()
    
    return _model_registry_instance


# Export
__all__ = ["ModelRegistry",
           "ModelUsageStats", 
           "get_model_registry"
          ]