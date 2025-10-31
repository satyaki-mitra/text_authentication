# DEPENDENCIES
import numpy as np
from typing import Any
from typing import List
from typing import Dict 
from loguru import logger
from typing import Optional
from dataclasses import dataclass
from config.settings import settings
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from sklearn.ensemble import RandomForestClassifier
from config.threshold_config import get_threshold_for_domain
from config.threshold_config import get_active_metric_weights


@dataclass
class EnsembleResult:
    """
    Result from ensemble classification
    """
    final_verdict      : str  # "AI-Generated", "Human-Written", or "Mixed"
    ai_probability     : float
    human_probability  : float
    mixed_probability  : float
    overall_confidence : float
    domain             : Domain
    metric_results     : Dict[str, MetricResult]
    metric_weights     : Dict[str, float]
    weighted_scores    : Dict[str, float]
    reasoning          : List[str]
    uncertainty_score  : float
    consensus_level    : float
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"final_verdict"        : self.final_verdict,
                "ai_probability"       : round(self.ai_probability, 4),
                "human_probability"    : round(self.human_probability, 4),
                "mixed_probability"    : round(self.mixed_probability, 4),
                "overall_confidence"   : round(self.overall_confidence, 4),
                "domain"               : self.domain.value,
                "uncertainty_score"    : round(self.uncertainty_score, 4),
                "consensus_level"      : round(self.consensus_level, 4),
                "metric_contributions" : {name: {"weight"         : round(self.metric_weights.get(name, 0.0), 4),
                                                 "weighted_score" : round(self.weighted_scores.get(name, 0.0), 4),
                                                 "ai_prob"        : round(result.ai_probability, 4),
                                                 "confidence"     : round(result.confidence, 4),
                                                }
                                                for name, result in self.metric_results.items()
                                        },
                "reasoning"            : self.reasoning,
               }


class EnsembleClassifier:
    """
    Eensemble classifier with multiple aggregation strategies
    
    Features:
    - Domain-aware dynamic weighting
    - Confidence-calibrated aggregation
    - Uncertainty quantification
    - Consensus analysis
    - Fallback strategies
    - Feature-based ML ensemble (optional)
    """
    def __init__(self, primary_method: str = "confidence_calibrated", fallback_method: str = "domain_weighted", use_ml_ensemble: bool = False, min_metrics_required: int = 3):
        """
        Initialize advanced ensemble classifier
        
        Arguments:
        ----------
            primary_method      : Primary aggregation method : "confidence_calibrated", "domain_adaptive", "consensus_based", "ml_ensemble"
            
            fallback_method     : Fallback method if primary fails : "domain_weighted", "confidence_weighted", "simple_average"
            
            use_ml_ensemble     : Use RandomForest for final aggregation (requires training)
            
            min_metrics_required: Minimum number of valid metrics required
        """
        self.primary_method       = primary_method
        self.fallback_method      = fallback_method
        self.use_ml_ensemble      = use_ml_ensemble
        self.min_metrics_required = min_metrics_required
        self.ml_model             = None
        
        logger.info(f"AdvancedEnsembleClassifier initialized (primary={primary_method}, fallback={fallback_method}, ml_ensemble={use_ml_ensemble})")
    

    def predict(self, metric_results: Dict[str, MetricResult], domain: Domain = Domain.GENERAL) -> EnsembleResult:
        """
        Combine metric results using advanced ensemble methods
        
        Arguments:
        ----------
            metric_results { dict }  : Dictionary mapping metric names to MetricResult objects

            domain        { Domain } : Text domain for adaptive thresholding
            
        Returns:
        --------
            { EnsembleResult }       : EnsembleResult object with final prediction
        """
        try:
            # Filter and validate metrics
            valid_results, validation_info = self._validate_metrics(metric_results)
            
            if (len(valid_results) < self.min_metrics_required):
                logger.warning(f"Insufficient valid metrics: {len(valid_results)}/{self.min_metrics_required}")
                return self._create_fallback_result(domain, metric_results, "insufficient_metrics")
            
            # Get domain-specific base weights
            enabled_metrics    = {name: True for name in valid_results.keys()}
            base_weights       = get_active_metric_weights(domain, enabled_metrics)
            
            # Try primary aggregation method : Initialize in case all methods fail unexpectedly
            calculated_weights = dict()
            aggregated         = {"ai_probability"    : 0.5, 
                                  "human_probability" : 0.5, 
                                  "mixed_probability" : 0.0,
                                 }

            try:
                if (self.primary_method == "confidence_calibrated"):
                    aggregated, calculated_weights = self._confidence_calibrated_aggregation(results      = valid_results, 
                                                                                             base_weights = base_weights,
                                                                                             domain       = domain,
                                                                                            )
                
                elif (self.primary_method == "domain_adaptive"):
                    aggregated, calculated_weights = self._domain_adaptive_aggregation(results      = valid_results, 
                                                                                       base_weights = base_weights, 
                                                                                       domain       = domain,
                                                                                      )
                
                elif (self.primary_method == "consensus_based"):
                    aggregated, calculated_weights = self._consensus_based_aggregation(results      = valid_results, 
                                                                                       base_weights = base_weights,
                                                                                       domain       = domain,
                                                                                      )
                
                elif ((self.primary_method == "ml_ensemble") and self.use_ml_ensemble):
                    aggregated, calculated_weights = self._ml_ensemble_aggregation(results      = valid_results, 
                                                                                   base_weights = base_weights,
                                                                                   domain       = domain,
                                                                                  )
                
                else:
                    # Fallback to domain weighted
                    aggregated, calculated_weights = self._domain_weighted_aggregation(results      = valid_results, 
                                                                                       base_weights = base_weights,
                                                                                       domain       = domain,
                                                                                      )
            
            except Exception as e:
                logger.warning(f"Primary aggregation failed: {e}, using fallback")
                aggregated, calculated_weights = self._apply_fallback_aggregation(results      = valid_results, 
                                                                                  base_weights = base_weights,
                                                                                 )
            
            # Start with the calculated weights (from valid_results)
            final_metric_weights = calculated_weights.copy() 

            # Iterate through the *original* metric_results input to the ensemble
            for original_metric_name in metric_results.keys():
                # If a metric from the original input wasn't included in calculated_weights :assign it a weight of 0.0.
                if original_metric_name not in final_metric_weights:
                    final_metric_weights[original_metric_name] = 0.0

            # Calculate advanced metrics using the CALCULATED weights (from valid_results), not the final ones
            overall_confidence   = self._calculate_advanced_confidence(results    = valid_results, 
                                                                       weights    = calculated_weights, 
                                                                       aggregated = aggregated,
                                                                      )

            uncertainty_score    = self._calculate_uncertainty(results    = valid_results, 
                                                               weights    = calculated_weights, 
                                                               aggregated = aggregated,
                                                              )

            consensus_level      = self._calculate_consensus_level(results = valid_results)
            
            # Apply domain-specific threshold with uncertainty consideration
            domain_thresholds    = get_threshold_for_domain(domain = domain)
            final_verdict        = self._apply_adaptive_threshold(aggregated     = aggregated, 
                                                                  base_threshold = domain_thresholds.ensemble_threshold,
                                                                  uncertainty    = uncertainty_score,
                                                                 )
            
            # Generate detailed reasoning using the CALCULATED weights
            reasoning            = self._generate_detailed_reasoning(results     = valid_results, 
                                                                     weights     = calculated_weights,
                                                                     aggregated  = aggregated, 
                                                                     verdict     = final_verdict, 
                                                                     uncertainty = uncertainty_score, 
                                                                     consensus   = consensus_level,
                                                                    )
            
            # Calculate weighted scores based on the CALCULATED weights (from valid_results)
            weighted_scores      = {name: result.ai_probability * calculated_weights.get(name, 0.0) for name, result in valid_results.items()}
            
            return EnsembleResult(final_verdict      = final_verdict,
                                  ai_probability     = aggregated["ai_probability"],
                                  human_probability  = aggregated["human_probability"],
                                  mixed_probability  = aggregated["mixed_probability"],
                                  overall_confidence = overall_confidence,
                                  domain             = domain,
                                  metric_results     = metric_results,
                                  metric_weights     = final_metric_weights,
                                  weighted_scores    = weighted_scores,
                                  reasoning          = reasoning,
                                  uncertainty_score  = uncertainty_score,
                                  consensus_level    = consensus_level,
                                 )
            
        except Exception as e:
            logger.error(f"Error in advanced ensemble prediction: {e}")
            return self._create_fallback_result(domain, metric_results, str(e))

    
    def _validate_metrics(self, results: Dict[str, MetricResult]) -> tuple:
        """
        Validate metrics and return quality information
        """
        valid_results   = dict()
        validation_info = {'failed_metrics'          : [],
                           'low_confidence_metrics'  : [],
                           'high_confidence_metrics' : [],
                          }
        
        for name, result in results.items():
            if result.error is not None:
                validation_info['failed_metrics'].append(name)
                continue
            
            if (result.confidence < 0.3):
                validation_info['low_confidence_metrics'].append(name)
                # Still include but with lower weight consideration
                valid_results[name] = result
            
            elif (result.confidence > 0.7):
                validation_info['high_confidence_metrics'].append(name)
                valid_results[name] = result
            
            else:
                valid_results[name] = result
        
        return valid_results, validation_info
    

    def _confidence_calibrated_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float], domain: Domain) -> tuple:
        """
        Confidence-calibrated aggregation with domain adaptation
        """
        # Calculate confidence-adjusted weights
        confidence_weights = dict()

        for name, result in results.items():
            base_weight              = base_weights.get(name, 0.0)
            # Confidence-based adjustment with non-linear scaling
            confidence_factor        = self._sigmoid_confidence_adjustment(confidence = result.confidence)
            confidence_weights[name] = base_weight * confidence_factor
        
        # Normalize weights
        total_weight = sum(confidence_weights.values())

        if (total_weight > 0):
            confidence_weights = {name: w / total_weight for name, w in confidence_weights.items()}
        
        # Domain-specific calibration
        domain_calibration = self._get_domain_calibration(domain = domain)
        calibrated_results = self._calibrate_probabilities(results     = results, 
                                                           calibration = domain_calibration,
                                                          )
        
        # Weighted aggregation
        return self._weighted_aggregation(calibrated_results, confidence_weights), confidence_weights
    

    def _domain_adaptive_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float], domain: Domain) -> tuple:
        """
        Domain-adaptive aggregation considering metric performance per domain
        """
        # Get domain-specific performance weights
        domain_weights = self._get_domain_performance_weights(domain, list(results.keys()))
        
        # Combine with base weights
        combined_weights = dict()
        for name in results.keys():
            domain_weight          = domain_weights.get(name, 1.0)
            base_weight            = base_weights.get(name, 0.0)
            combined_weights[name] = base_weight * domain_weight
        
        # Normalize
        total_weight = sum(combined_weights.values())
        if (total_weight > 0):
            combined_weights = {name: w / total_weight for name, w in combined_weights.items()}
        
        return self._weighted_aggregation(results, combined_weights), combined_weights
    

    def _consensus_based_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> tuple:
        """
        Consensus-based aggregation that rewards metric agreement
        """
        # Calculate consensus scores
        consensus_weights = self._calculate_consensus_weights(results, base_weights)
        
        aggregations      = self._weighted_aggregation(results = results, 
                                                       weights = consensus_weights,
                                                      )
        return aggregations, consensus_weights
    

    def _ml_ensemble_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> tuple:
        """
        Machine learning-based ensemble aggregation
        """
        if self.ml_model is None:
            logger.warning("ML model not available, falling back to weighted average")
            return self._weighted_aggregation(results, base_weights), base_weights
        
        try:
            # Extract features from metric results
            features   = self._extract_ml_features(results = results)
            
            # Predict using ML model
            prediction = self.ml_model.predict_proba([features])[0]
            
            # For now, assume binary classification [human_prob, ai_prob]
            if (len(prediction) == 2):
                ai_prob, human_prob = prediction[1], prediction[0]
                mixed_prob          = 0.0

            else:
                # Multi-class - adjust accordingly
                ai_prob, human_prob, mixed_prob = prediction
            
            aggregated = {"ai_probability"    : ai_prob,
                          "human_probability" : human_prob,
                          "mixed_probability" : mixed_prob,
                         }
            
            return aggregated, base_weights
            
        except Exception as e:
            logger.warning(f"ML ensemble failed: {e}, using fallback")
            return self._weighted_aggregation(results, base_weights), base_weights
    

    def _domain_weighted_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> tuple:
        """
        Simple domain-weighted aggregation (fallback method)
        """
        return self._weighted_aggregation(results, base_weights), base_weights
    

    def _apply_fallback_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> tuple:
        """
        Apply fallback aggregation method
        """
        if (self.fallback_method == "confidence_weighted"):
            return self._confidence_weighted_aggregation(results), base_weights
        
        elif (self.fallback_method == "simple_average"):
            return self._simple_average_aggregation(results), base_weights

        else:
            return self._domain_weighted_aggregation(results, base_weights), base_weights
    

    def _weighted_aggregation(self, results: Dict[str, MetricResult], weights: Dict[str, float]) -> Dict[str, float]:
        """
        Core weighted aggregation logic
        """
        ai_scores    = list()
        human_scores = list()
        mixed_scores = list()
        total_weight = 0.0
        
        for name, result in results.items():
            weight = weights.get(name, 0.0)
            
            if (weight > 0):
                ai_scores.append(result.ai_probability * weight)
                human_scores.append(result.human_probability * weight)
                mixed_scores.append(result.mixed_probability * weight)
                
                total_weight += weight
        
        if (total_weight == 0):
            return {"ai_probability"    : 0.5, 
                    "human_probability" : 0.5, 
                    "mixed_probability" : 0.0,
                   }
        
        # Calculate weighted averages
        ai_prob    = sum(ai_scores) / total_weight
        human_prob = sum(human_scores) / total_weight
        mixed_prob = sum(mixed_scores) / total_weight
        
        # Normalize
        total     = ai_prob + human_prob + mixed_prob
        
        if (total > 0):
            ai_prob    /= total
            human_prob /= total
            mixed_prob /= total
        
        return {"ai_probability"    : ai_prob, 
                "human_probability" : human_prob,
                "mixed_probability" : mixed_prob,
               }
    

    def _confidence_weighted_aggregation(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Confidence-weighted aggregation
        """
        return self._weighted_aggregation(results, {name: result.confidence for name, result in results.items()})
    

    def _simple_average_aggregation(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Simple average aggregation
        """
        return self._weighted_aggregation(results, {name: 1.0 for name in results.keys()})
    

    def _sigmoid_confidence_adjustment(self, confidence: float) -> float:
        """
        Non-linear confidence adjustment using sigmoid
        """
        # Sigmoid that emphasizes differences around 0.5 confidence
        return 1.0 / (1.0 + np.exp(-10.0 * (confidence - 0.5)))
    

    def _get_domain_calibration(self, domain: Domain) -> Dict[str, float]:
        """
        Get domain-specific calibration factors
        """
        # This would typically come from validation data
        # For now, return neutral calibration : FUTURE WQORK
        return {}
    

    def _calibrate_probabilities(self, results: Dict[str, MetricResult], calibration: Dict[str, float]) -> Dict[str, MetricResult]:
        """
        Calibrate probabilities based on domain performance
        """
        calibrated = dict()
        for name, result in results.items():
            cal_factor       = calibration.get(name, 1.0)
            # Simple calibration - could be more sophisticated
            new_ai_prob      = min(1.0, max(0.0, result.ai_probability * cal_factor))
            calibrated[name] = MetricResult(metric_name       = result.metric_name,
                                            ai_probability    = new_ai_prob,
                                            human_probability = 1.0 - new_ai_prob,  # Simplified
                                            mixed_probability = result.mixed_probability,
                                            confidence        = result.confidence,
                                            details           = result.details
                                           )
        return calibrated
    

    def _get_domain_performance_weights(self, domain: Domain, metric_names: List[str]) -> Dict[str, float]:
        """
        Get domain-specific performance weights (would come from validation data)
        """
        # Placeholder - in practice, this would be based on historical performance per domain : FUTURE WORK
        performance_weights = {'structural'                   : 1.0, 
                               'entropy'                      : 1.0, 
                               'semantic_analysis'            : 1.0,
                               'linguistic'                   : 1.0, 
                               'perplexity'                   : 1.0, 
                               'multi_perturbation_stability' : 1.0,
                              }
        
        # Domain-specific adjustments for all 16 domains
        domain_adjustments  = {Domain.GENERAL       : {'structural'                   : 1.0,
                                                       'perplexity'                   : 1.0,
                                                       'entropy'                      : 1.0,
                                                       'semantic_analysis'            : 1.0,
                                                       'linguistic'                   : 1.0,
                                                       'multi_perturbation_stability' : 1.0,
                                                      },
                               Domain.ACADEMIC      : {'structural'                   : 1.2, 
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.3, 
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.CREATIVE      : {'structural'                   : 0.9,
                                                       'perplexity'                   : 1.1,
                                                       'entropy'                      : 1.2, 
                                                       'semantic_analysis'            : 1.0,
                                                       'linguistic'                   : 1.1,
                                                       'multi_perturbation_stability' : 0.9,
                                                      },
                               Domain.AI_ML         : {'structural'                   : 1.2,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.SOFTWARE_DEV  : {'structural'                   : 1.2,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.TECHNICAL_DOC : {'structural'                   : 1.3, 
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.2,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.ENGINEERING   : {'structural'                   : 1.2,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.SCIENCE       : {'structural'                   : 1.2,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.BUSINESS      : {'structural'                   : 1.1,
                                                       'perplexity'                   : 1.2,
                                                       'entropy'                      : 1.0,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.1,
                                                       'multi_perturbation_stability' : 0.9,
                                                      },
                               Domain.LEGAL         : {'structural'                   : 1.3,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.2,
                                                       'linguistic'                   : 1.3,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.MEDICAL       : {'structural'                   : 1.2,
                                                       'perplexity'                   : 1.3,
                                                       'entropy'                      : 0.9,
                                                       'semantic_analysis'            : 1.2,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.JOURNALISM    : {'structural'                   : 1.1,
                                                       'perplexity'                   : 1.2,
                                                       'entropy'                      : 1.0,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.1,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.MARKETING     : {'structural'                   : 1.0,
                                                       'perplexity'                   : 1.1,
                                                       'entropy'                      : 1.1,
                                                       'semantic_analysis'            : 1.0,
                                                       'linguistic'                   : 1.2,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.SOCIAL_MEDIA  : {'structural'                   : 0.8,
                                                       'perplexity'                   : 1.0,
                                                       'entropy'                      : 1.3, 
                                                       'semantic_analysis'            : 0.9,
                                                       'linguistic'                   : 0.7,
                                                       'multi_perturbation_stability' : 0.9,
                                                      },
                               Domain.BLOG_PERSONAL : {'structural'                   : 0.9,
                                                       'perplexity'                   : 1.1,
                                                       'entropy'                      : 1.2,
                                                       'semantic_analysis'            : 1.0,
                                                       'linguistic'                   : 1.0,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                               Domain.TUTORIAL      : {'structural'                   : 1.1,
                                                       'perplexity'                   : 1.2,
                                                       'entropy'                      : 1.0,
                                                       'semantic_analysis'            : 1.1,
                                                       'linguistic'                   : 1.1,
                                                       'multi_perturbation_stability' : 0.8,
                                                      },
                              }
        
        adjustments         = domain_adjustments.get(domain, {})

        return {name: performance_weights.get(name, 1.0) * adjustments.get(name, 1.0) for name in metric_names}
    

    def _calculate_consensus_weights(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights based on metric consensus
        """
        # Calculate average AI probability
        avg_ai_prob       = np.mean([r.ai_probability for r in results.values()])
        
        consensus_weights = dict()

        for name, result in results.items():
            base_weight             = base_weights.get(name, 0.0)
            # Reward metrics that agree with consensus
            agreement               = 1.0 - abs(result.ai_probability - avg_ai_prob)
            consensus_weights[name] = base_weight * (0.5 + 0.5 * agreement)  # 0.5-1.0 range
        
        # Normalize
        total_weight = sum(consensus_weights.values())
        if (total_weight > 0):
            consensus_weights = {name: w / total_weight for name, w in consensus_weights.items()}
        
        return consensus_weights
    

    def _extract_ml_features(self, results: Dict[str, MetricResult]) -> List[float]:
        """
        Extract features for ML ensemble
        """
        features = list()
        for name in sorted(results.keys()):  # Ensure consistent order
            result = results[name]
            features.extend([result.ai_probability,
                             result.human_probability,
                             result.mixed_probability,
                             result.confidence
                           ])

        return features
    

    def _calculate_advanced_confidence(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float]) -> float:
        """
        Calculate advanced confidence considering multiple factors
        """
        # Base confidence from metric confidences
        base_confidence         = sum(result.confidence * weights.get(name, 0.0) for name, result in results.items())
        
        # Agreement factor
        ai_probs                = [r.ai_probability for r in results.values()]
        agreement               = 1.0 - min(1.0, np.std(ai_probs) * 2.0)  # 0-1 scale
        
        # Certainty factor (how far from 0.5)
        certainty               = 1.0 - 2.0 * abs(aggregated["ai_probability"] - 0.5)
        
        # Metric quality factor
        high_confidence_metrics = sum(1 for r in results.values() if r.confidence > 0.7)
        quality_factor          = high_confidence_metrics / len(results) if results else 0.0
        
        # Combined confidence
        confidence              = (base_confidence * 0.4 + agreement * 0.3 + certainty * 0.2 + quality_factor * 0.1)
        
        return max(0.0, min(1.0, confidence))
    

    def _calculate_uncertainty(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float]) -> float:
        """
        Calculate uncertainty score
        """
        # Variance in predictions
        ai_probs               = [r.ai_probability for r in results.values()]
        variance_uncertainty   = np.var(ai_probs) if len(ai_probs) > 1 else 0.0
        
        # Confidence uncertainty
        avg_confidence         = np.mean([r.confidence for r in results.values()])
        confidence_uncertainty = 1.0 - avg_confidence
        
        # Decision uncertainty (how close to 0.5)
        decision_uncertainty   = 1.0 - 2.0 * abs(aggregated["ai_probability"] - 0.5)
        
        # Combined uncertainty
        uncertainty            = (variance_uncertainty * 0.4 + confidence_uncertainty * 0.3 + decision_uncertainty * 0.3)
        
        return max(0.0, min(1.0, uncertainty))
    

    def _calculate_consensus_level(self, results: Dict[str, MetricResult]) -> float:
        """
        Calculate consensus level among metrics
        """
        if (len(results) < 2):
            # Perfect consensus with only one metric
            return 1.0  
        
        ai_probs  = [r.ai_probability for r in results.values()]
        std_dev   = np.std(ai_probs)
        
        # Convert to consensus level (1.0 = perfect consensus, 0.0 = no consensus)
        consensus = 1.0 - min(1.0, std_dev * 2.0)
        
        return consensus
    

    def _apply_adaptive_threshold(self, aggregated: Dict[str, float], base_threshold: float, uncertainty: float) -> str:
        """
        Apply adaptive threshold considering uncertainty
        """
        ai_prob            = aggregated.get("ai_probability", 0.5)
        mixed_prob         = aggregated.get("mixed_probability", 0.0)
        
        # Adjust threshold based on uncertainty : Higher uncertainty requires more confidence
        adjusted_threshold = base_threshold + (uncertainty * 0.1) 
        
        # Check for mixed content
        if ((mixed_prob > 0.25) or ((uncertainty > 0.6) and (0.3 < ai_prob < 0.7))):
            return "Mixed (AI + Human)"
        
        # Apply adjusted threshold
        if (ai_prob >= adjusted_threshold):
            return "AI-Generated"

        elif (ai_prob <= (1.0 - adjusted_threshold)):
            return "Human-Written"

        else:
            return "Uncertain"
    

    def _generate_detailed_reasoning(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float], 
                                     verdict: str, uncertainty: float, consensus: float) -> List[str]:
        """
        Generate detailed reasoning for the prediction
        """
        reasoning = list()
        
        # Overall assessment
        ai_prob    = aggregated.get("ai_probability", 0.5)
        mixed_prob = aggregated.get("mixed_probability", 0.0)

        reasoning.append(f"## Ensemble Analysis Result")
        reasoning.append(f"**Final Verdict**: {verdict}")
        reasoning.append(f"**AI Probability**: {ai_prob:.1%}")
        reasoning.append(f"**Confidence Level**: {self._get_confidence_label(ai_prob)}")
        reasoning.append(f"**Uncertainty**: {uncertainty:.1%}")
        reasoning.append(f"**Consensus**: {consensus:.1%}")
        
        # Metric analysis
        reasoning.append(f"\n## Metric Analysis")
        
        sorted_metrics = sorted(results.items(), key=lambda x: weights.get(x[0], 0.0), reverse=True)
        
        for name, result in sorted_metrics:
            weight       = weights.get(name, 0.0)
            contribution = "High" if (weight > 0.15) else "Medium" if (weight > 0.08) else "Low"
            
            reasoning.append(f"**{name}**: {result.ai_probability:.1%} AI "
                             f"(Confidence: {result.confidence:.1%}, "
                             f"Contribution: {contribution})")
        
        # Key factors
        reasoning.append(f"\n## Key Decision Factors")
        
        if (uncertainty > 0.7):
            reasoning.append("⚠ **High uncertainty** - Metrics show significant disagreement")
        
        elif (consensus > 0.8):
            reasoning.append("✓ **Strong consensus** - All metrics agree on classification")
        
        top_metric = sorted_metrics[0] if sorted_metrics else None
        
        if (top_metric and (weights.get(top_metric[0], 0.0) > 0.2)):
            reasoning.append(f"🎯 **Dominant metric** - {top_metric[0]} had strongest influence")
        
        if (mixed_prob > 0.2):
            reasoning.append("🔀 **Mixed signals** - Content shows characteristics of both AI and human writing")
        
        return reasoning
    

    def _get_confidence_label(self, ai_prob: float) -> str:
        """
        Get human-readable confidence label
        """
        if ((ai_prob > 0.9) or (ai_prob < 0.1)):
            return "Very High"

        elif ((ai_prob > 0.8) or (ai_prob < 0.2)):
            return "High"

        elif ((ai_prob > 0.7) or (ai_prob < 0.3)):
            return "Moderate"

        else:
            return "Low"
    

    def _create_fallback_result(self, domain: Domain, metric_results: Dict[str, MetricResult], error: str) -> EnsembleResult:
        """
        Create fallback result when ensemble cannot make a confident decision
        """
        return EnsembleResult(final_verdict      = "Uncertain",
                              ai_probability     = 0.5,
                              human_probability  = 0.5,
                              mixed_probability  = 0.0,
                              overall_confidence = 0.0,
                              domain             = domain,
                              metric_results     = metric_results,
                              metric_weights     = {},
                              weighted_scores    = {},
                              reasoning          = [f"Ensemble analysis inconclusive", f"Reason: {error}"],
                              uncertainty_score  = 1.0,
                              consensus_level    = 0.0,
                             )


# Export
__all__ = ["EnsembleResult",
           "EnsembleClassifier",
          ]