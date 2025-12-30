# DEPENDENCIES
import numpy as np
from typing import List
from typing import Dict 
from loguru import logger
from config.enums import Domain
from config.schemas import MetricResult
from config.schemas import EnsembleResult
from config.constants import metrics_ensemble_params
from config.threshold_config import get_threshold_for_domain
from config.threshold_config import get_active_metric_weights


class EnsembleClassifier:
    """
    Ensemble classifier with multiple aggregation strategies
    
    Features:
    - Domain-aware dynamic weighting
    - Confidence-calibrated aggregation
    - Uncertainty quantification
    - Consensus analysis
    - Fallback strategies
    """
    def __init__(self, primary_method: str = "confidence_calibrated", fallback_method: str = "domain_weighted", min_metrics_required: int = None, execution_mode = "parallel"):
        """
        Initialize advanced ensemble classifier
        
        Arguments:
        ----------
            primary_method       : Primary aggregation method : "confidence_calibrated", "consensus_based"
            
            fallback_method      : Fallback method if primary fails : "domain_weighted", "confidence_weighted", "simple_average"
            
            min_metrics_required : Minimum number of valid metrics required (overrides default)
        """
        self.primary_method       = primary_method
        self.fallback_method      = fallback_method
        self.min_metrics_required = min_metrics_required or metrics_ensemble_params.MIN_METRICS_REQUIRED
        self.params               = metrics_ensemble_params
        self.execution_mode       = execution_mode
        
        logger.info(f"EnsembleClassifier initialized (primary={primary_method}, fallback={fallback_method})")
    

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
            # Filter out metrics with errors
            valid_results = self._filter_valid_metrics(results = metric_results)
            
            if (len(valid_results) < self.min_metrics_required):
                logger.warning(f"Insufficient valid metrics: {len(valid_results)}/{self.min_metrics_required}")
                return self._create_fallback_result(domain, metric_results, "insufficient_metrics")
            
            # Get domain-specific base weights
            enabled_metrics    = {name: True for name in valid_results.keys()}
            base_weights       = get_active_metric_weights(domain, enabled_metrics)
            
            # Try primary aggregation method
            calculated_weights = dict()
            aggregated         = {"synthetic_probability"    : self.params.DEFAULT_SYNTHETIC_PROB, 
                                  "authentic_probability"    : self.params.DEFAULT_AUTHENTIC_PROB, 
                                  "hybrid_probability"       : self.params.DEFAULT_HYBRID_PROB,
                                 }

            try:
                if (self.primary_method == "confidence_calibrated"):
                    aggregated, calculated_weights = self._confidence_calibrated_aggregation(results      = valid_results, 
                                                                                             base_weights = base_weights,
                                                                                             domain       = domain,
                                                                                            )
                
                elif (self.primary_method == "consensus_based"):
                    aggregated, calculated_weights = self._consensus_based_aggregation(results      = valid_results, 
                                                                                       base_weights = base_weights,
                                                                                      )
                
                else:
                    # Fallback to domain weighted
                    aggregated, calculated_weights = self._domain_weighted_aggregation(results      = valid_results, 
                                                                                       base_weights = base_weights,
                                                                                      )
            
            except Exception as e:
                logger.warning(f"Primary aggregation failed: {e}, using fallback")
                aggregated, calculated_weights = self._apply_fallback_aggregation(results      = valid_results, 
                                                                                  base_weights = base_weights,
                                                                                 )
            
            # Start with the calculated weights (from valid_results)
            final_metric_weights = calculated_weights.copy() 

            # Assign zero weight to any original metrics that weren't included in valid_results
            for original_metric_name in metric_results.keys():
                if (original_metric_name not in final_metric_weights):
                    final_metric_weights[original_metric_name] = 0.0

            # Calculate advanced metrics
            overall_confidence   = self._calculate_confidence(results    = valid_results, 
                                                              weights    = calculated_weights, 
                                                              aggregated = aggregated,
                                                             )

            uncertainty_score    = self._calculate_uncertainty(results    = valid_results, 
                                                               aggregated = aggregated,
                                                              )

            consensus_level      = self._calculate_consensus_level(results = valid_results)
            
            # Apply domain-specific threshold with uncertainty consideration
            domain_thresholds    = get_threshold_for_domain(domain = domain)
            final_verdict        = self._apply_adaptive_threshold(aggregated     = aggregated, 
                                                                  base_threshold = domain_thresholds.ensemble_threshold,
                                                                  uncertainty    = uncertainty_score,
                                                                 )
            
            # Generate reasoning
            reasoning            = self._generate_reasoning(results     = valid_results, 
                                                           weights     = calculated_weights,
                                                           aggregated  = aggregated, 
                                                           verdict     = final_verdict, 
                                                           uncertainty = uncertainty_score, 
                                                           consensus   = consensus_level,
                                                          )
            
            # Calculate weighted scores
            weighted_scores      = {name: result.synthetic_probability * calculated_weights.get(name, 0.0) 
                                   for name, result in valid_results.items()}
            
            return EnsembleResult(final_verdict         = final_verdict,
                                  synthetic_probability = aggregated["synthetic_probability"],
                                  authentic_probability = aggregated["authentic_probability"],
                                  hybrid_probability    = aggregated["hybrid_probability"],
                                  overall_confidence    = overall_confidence,
                                  domain                = domain,
                                  metric_results        = metric_results,
                                  metric_weights        = final_metric_weights,
                                  weighted_scores       = weighted_scores,
                                  reasoning             = reasoning,
                                  uncertainty_score     = uncertainty_score,
                                  consensus_level       = consensus_level,
                                  execution_mode        = self.execution_mode,
                                 )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return self._create_fallback_result(domain, metric_results, str(e))

    
    def _filter_valid_metrics(self, results: Dict[str, MetricResult]) -> Dict[str, MetricResult]:
        """
        Filter out failed metrics (error != None).
        Confidence is handled during aggregation, not validation.
        """
        valid_results = dict()
        
        for name, result in results.items():
            if result.error is not None:
                continue
            
            valid_results[name] = result
        
        return valid_results
    

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
        confidence_weights = self._normalize_weights(confidence_weights)
        
        # Domain-specific calibration
        domain_calibration = self._get_domain_calibration(domain = domain)
        calibrated_results = self._calibrate_probabilities(results     = results, 
                                                           calibration = domain_calibration,
                                                          )
        
        # Weighted aggregation
        return self._weighted_aggregation(calibrated_results, confidence_weights), confidence_weights
    

    def _consensus_based_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> tuple:
        """
        Consensus-based aggregation that rewards metric agreement
        """
        # Calculate consensus scores
        consensus_weights = self._calculate_consensus_weights(results      = results, 
                                                              base_weights = base_weights,
                                                             )
        
        consensus_weights = self._normalize_weights(consensus_weights)
        
        aggregations      = self._weighted_aggregation(results = results, 
                                                       weights = consensus_weights,
                                                      )
        return aggregations, consensus_weights
    

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
            return self._confidence_weighted_aggregation(results = results), base_weights
        
        elif (self.fallback_method == "simple_average"):
            return self._simple_average_aggregation(results = results), base_weights

        else:
            return self._domain_weighted_aggregation(results = results, base_weights = base_weights), base_weights
    

    def _weighted_aggregation(self, results: Dict[str, MetricResult], weights: Dict[str, float]) -> Dict[str, float]:
        """
        Core weighted aggregation logic
        """
        synthetic_scores = list()
        authentic_scores = list()
        hybrid_scores    = list()
        total_weight     = 0.0
        
        for name, result in results.items():
            weight = weights.get(name, 0.0)
            
            if (weight > 0):
                synthetic_scores.append(result.synthetic_probability * weight)
                authentic_scores.append(result.authentic_probability * weight)
                hybrid_scores.append(result.hybrid_probability * weight)
                
                total_weight += weight
        
        if (total_weight == 0):
            return {"synthetic_probability" : self.params.DEFAULT_SYNTHETIC_PROB, 
                    "authentic_probability" : self.params.DEFAULT_AUTHENTIC_PROB, 
                    "hybrid_probability"    : self.params.DEFAULT_HYBRID_PROB,
                   }
        
        # Calculate weighted averages
        synthetic_prob = sum(synthetic_scores) / total_weight
        authentic_prob = sum(authentic_scores) / total_weight
        hybrid_prob    = sum(hybrid_scores) / total_weight
        
        # Normalize probabilities to sum to 1.0
        total          = synthetic_prob + authentic_prob + hybrid_prob
        
        if (total > 0):
            synthetic_prob /= total
            authentic_prob /= total
            hybrid_prob    /= total
        
        return {"synthetic_probability" : synthetic_prob, 
                "authentic_probability" : authentic_prob,
                "hybrid_probability"    : hybrid_prob,
               }
    

    def _confidence_weighted_aggregation(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Confidence-weighted aggregation
        """
        weights = {name: result.confidence for name, result in results.items()}
        weights = self._normalize_weights(weights)
        return self._weighted_aggregation(results, weights)
    

    def _simple_average_aggregation(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """
        Simple average aggregation
        """
        return self._weighted_aggregation(results, {name: 1.0 for name in results.keys()})
    

    def _sigmoid_confidence_adjustment(self, confidence: float) -> float:
        """
        Non-linear confidence adjustment using sigmoid
        """
        # Sigmoid that emphasizes differences around the center
        return 1.0 / (1.0 + np.exp(-self.params.SIGMOID_CONFIDENCE_SCALE * (confidence - self.params.SIGMOID_CENTER)))
    

    def _get_domain_calibration(self, domain: Domain) -> Dict[str, float]:
        """
        Get domain-specific calibration factors
        """
        # This would typically come from validation data
        # For now, return neutral calibration
        return {}
    

    def _calibrate_probabilities(self, results: Dict[str, MetricResult], calibration: Dict[str, float]) -> Dict[str, MetricResult]:
        """
        Calibrate probabilities based on domain performance
        """
        calibrated = dict()

        for name, result in results.items():
            cal_factor         = calibration.get(name, 1.0)
            # Simple calibration
            new_synthetic_prob = min(1.0, max(0.0, result.synthetic_probability * cal_factor))

            calibrated[name]   = MetricResult(metric_name           = result.metric_name,
                                              synthetic_probability = new_synthetic_prob,
                                              authentic_probability = 1.0 - new_synthetic_prob,
                                              hybrid_probability    = result.hybrid_probability,
                                              confidence            = result.confidence,
                                              details               = result.details
                                             )
        return calibrated
    

    def _calculate_consensus_weights(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights based on metric consensus
        """
        # Calculate average synthetic probability
        avg_synthetic_prob = np.mean([r.synthetic_probability for r in results.values()])
        
        consensus_weights  = dict()

        for name, result in results.items():
            base_weight             = base_weights.get(name, 0.0)
            # Reward metrics that agree with consensus
            agreement               = 1.0 - abs(result.synthetic_probability - avg_synthetic_prob)
            consensus_weights[name] = base_weight * (0.5 + 0.5 * agreement)  # 0.5-1.0 range
        
        return consensus_weights
    

    def _calculate_confidence(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float]) -> float:
        """
        Calculate confidence considering multiple factors
        """
        # Base confidence from metric confidences
        base_confidence         = sum(result.confidence * weights.get(name, 0.0) for name, result in results.items())
        
        # Agreement factor
        synthetic_probs         = [r.synthetic_probability for r in results.values()]
        agreement               = 1.0 - min(1.0, np.std(synthetic_probs) * self.params.CONSENSUS_STD_SCALING)
        
        # Certainty factor (how far from 0.5)
        certainty               = 1.0 - 2.0 * abs(aggregated["synthetic_probability"] - 0.5)
        
        # Metric quality factor
        high_confidence_metrics = sum(1 for r in results.values() if r.confidence > self.params.HIGH_CONFIDENCE_THRESHOLD)
        quality_factor          = high_confidence_metrics / len(results) if results else 0.0
        
        # Combined confidence
        confidence              = (base_confidence * self.params.CONFIDENCE_WEIGHT_BASE + 
                                   agreement * self.params.CONFIDENCE_WEIGHT_AGREEMENT + 
                                   certainty * self.params.CONFIDENCE_WEIGHT_CERTAINTY + 
                                   quality_factor * self.params.CONFIDENCE_WEIGHT_QUALITY)
        
        return max(0.0, min(1.0, confidence))
    

    def _calculate_uncertainty(self, results: Dict[str, MetricResult], aggregated: Dict[str, float]) -> float:
        """
        Calculate uncertainty score
        """
        # Variance in predictions
        synthetic_probs        = [r.synthetic_probability for r in results.values()]
        variance_uncertainty   = np.var(synthetic_probs) if len(synthetic_probs) > 1 else 0.0
        
        # Confidence uncertainty
        avg_confidence         = np.mean([r.confidence for r in results.values()])
        confidence_uncertainty = 1.0 - avg_confidence
        
        # Decision uncertainty (how close to 0.5)
        decision_uncertainty   = 1.0 - 2.0 * abs(aggregated["synthetic_probability"] - 0.5)
        
        # Combined uncertainty
        uncertainty            = (variance_uncertainty * self.params.UNCERTAINTY_WEIGHT_VARIANCE + 
                                  confidence_uncertainty * self.params.UNCERTAINTY_WEIGHT_CONFIDENCE + 
                                  decision_uncertainty * self.params.UNCERTAINTY_WEIGHT_DECISION)
        
        return max(0.0, min(1.0, uncertainty))
    

    def _calculate_consensus_level(self, results: Dict[str, MetricResult]) -> float:
        """
        Calculate consensus level among metrics
        """
        if (len(results) < 2):
            # Perfect consensus with only one metric
            return 1.0  
        
        synthetic_probs  = [r.synthetic_probability for r in results.values()]
        std_dev          = np.std(synthetic_probs)
        
        # Convert to consensus level (1.0 = perfect consensus, 0.0 = no consensus)
        consensus        = 1.0 - min(1.0, std_dev * self.params.CONSENSUS_STD_SCALING)
        
        return consensus
    

    def _apply_adaptive_threshold(self, aggregated: Dict[str, float], base_threshold: float, uncertainty: float) -> str:
        """
        Apply adaptive threshold considering uncertainty
        """
        synthetic_prob      = aggregated.get("synthetic_probability", self.params.DEFAULT_SYNTHETIC_PROB)
        hybrid_prob         = aggregated.get("hybrid_probability", self.params.DEFAULT_HYBRID_PROB)
        
        # Adjust threshold based on uncertainty : Higher uncertainty requires more confidence
        adjusted_threshold = base_threshold + (uncertainty * self.params.UNCERTAINTY_THRESHOLD_ADJUSTMENT) 
        
        # Check for hybrid content
        # Case 1: Explicit hybrid probability from metrics
        # Case 2: High uncertainty + ambiguous synthetic score
        if ((hybrid_prob > self.params.HYBRID_PROB_THRESHOLD) or ((uncertainty > self.params.HYBRID_UNCERTAINTY_THRESHOLD) and (self.params.HYBRID_SYNTHETIC_RANGE_LOW < synthetic_prob < self.params.HYBRID_SYNTHETIC_RANGE_HIGH))):
            return "Hybrid"
        
        # Apply adjusted threshold
        if (synthetic_prob >= adjusted_threshold):
            return "Synthetically-Generated"

        elif (synthetic_prob <= (1.0 - adjusted_threshold)):
            return "Authentically-Written"

        else:
            return "Uncertain"
    

    def _generate_reasoning(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float], verdict: str, uncertainty: float, consensus: float) -> List[str]:
        """
        Generate reasoning for the prediction
        """
        reasoning      = list()
        
        # Overall assessment
        synthetic_prob = aggregated.get("synthetic_probability", self.params.DEFAULT_SYNTHETIC_PROB)
        hybrid_prob    = aggregated.get("hybrid_probability", self.params.DEFAULT_HYBRID_PROB)

        reasoning.append(f"## Ensemble Analysis Result")
        reasoning.append(f"**Final Verdict**: {verdict}")
        reasoning.append(f"**Synthetic Probability**: {synthetic_prob:.1%}")
        reasoning.append(f"**Confidence Level**: {self._get_confidence_label(synthetic_prob)}")
        reasoning.append(f"**Uncertainty**: {uncertainty:.1%}")
        reasoning.append(f"**Consensus**: {consensus:.1%}")
        
        # Metric analysis
        reasoning.append(f"\n## Metric Analysis")
        
        sorted_metrics = sorted(results.items(), key=lambda x: weights.get(x[0], 0.0), reverse=True)
        
        for name, result in sorted_metrics:
            weight       = weights.get(name, 0.0)
            
            if (weight > self.params.CONTRIBUTION_HIGH):
                contribution = "High"
            
            elif (weight > self.params.CONTRIBUTION_MEDIUM):
                contribution = "Medium"
            
            else:
                contribution = "Low"
            
            reasoning.append(f"**{name}**: {result.synthetic_probability:.1%} synthetic probability "
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
        
        if (hybrid_prob > self.params.HYBRID_PROB_THRESHOLD):
            reasoning.append("🔀 **Mixed signals** - Content shows characteristics of both synthetic and authentic writing")
        
        return reasoning
    

    def _get_confidence_label(self, synthetic_prob: float) -> str:
        """
        Get human-readable confidence label based on distance from decision boundaries
        """
        # Very high confidence: very clear synthetic or very clear authentic
        if ((synthetic_prob > 0.9) or (synthetic_prob < 0.1)):
            return "Very High"
        
        # High confidence: strongly synthetic or strongly authentic
        elif ((synthetic_prob > 0.8) or (synthetic_prob < 0.2)):
            return "High"
        
        # Moderate confidence: leaning synthetic or leaning authentic
        elif ((synthetic_prob > 0.7) or (synthetic_prob < 0.3)):
            return "Moderate"
        
        # Low confidence: close to decision boundary
        else:
            return "Low"
    

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0
        """
        total = sum(weights.values())
        
        if (total > 0):
            return {k: v / total for k, v in weights.items()}
            
        return weights
    

    def _create_fallback_result(self, domain: Domain, metric_results: Dict[str, MetricResult], error: str) -> EnsembleResult:
        """
        Create fallback result when ensemble cannot make a confident decision
        """
        return EnsembleResult(final_verdict         = "Uncertain",
                              synthetic_probability = self.params.DEFAULT_SYNTHETIC_PROB,
                              authentic_probability = self.params.DEFAULT_AUTHENTIC_PROB,
                              hybrid_probability    = self.params.DEFAULT_HYBRID_PROB,
                              overall_confidence    = 0.0,
                              domain                = domain,
                              metric_results        = metric_results,
                              metric_weights        = {},
                              weighted_scores       = {},
                              reasoning             = [f"Ensemble analysis inconclusive", f"Reason: {error}"],
                              uncertainty_score     = 1.0,
                              consensus_level       = 0.0,
                              execution_mode        = self.execution_mode,
                             )


# Export
__all__ = ["EnsembleClassifier"]