# DEPENDENCIES
import numpy as np
from typing import List
from typing import Dict 
from typing import Tuple
from loguru import logger
from config.enums import Domain
from config.schemas import MetricResult
from config.schemas import EnsembleResult
from config.threshold_config import get_threshold_for_domain
from config.threshold_config import get_active_metric_weights
from config.constants import metrics_ensemble_params as params


class EnsembleClassifier:
    """
    Ensemble classifier with domain-aware confidence-calibrated aggregation
    
    Features:
    - Domain-aware dynamic weighting
    - Power-based probability calibration (fixed from temperature scaling)
    - Uncertainty quantification
    - Consensus analysis
    """
    def __init__(self, calibration_temperature: float = 1.3, min_metrics_required: int = None, execution_mode: str = "sequential"):
        """
        Initialize ensemble classifier
        
        Arguments:
        ----------
            calibration_temperature { float } : Calibration strength (1.0-3.0)
                                                T > 1.0: softer probabilities (less confident)
                                                T < 1.0: sharper probabilities (more confident)
            
            min_metrics_required     { int }  : Minimum number of valid metrics required (default: 3)

            execution_mode           { str }  : Mode of execution: "sequential" or "parallel"
        """
        self.min_metrics_required    = min_metrics_required or params.MIN_METRICS_REQUIRED
        self.execution_mode          = execution_mode
        
        # Clamp calibration temperature to safe range
        self.calibration_temp        = np.clip(a     = calibration_temperature, 
                                               a_min = params.CALIBRATION_TEMP_MIN, 
                                               a_max = params.CALIBRATION_TEMP_MAX,
                                              )
        
        logger.info(f"EnsembleClassifier initialized (calibration_temp={self.calibration_temp}, min_metrics={self.min_metrics_required})")
    

    def predict(self, metric_results: Dict[str, MetricResult], domain: Domain = Domain.GENERAL) -> EnsembleResult:
        """
        Combine metric results using confidence-calibrated aggregation with probability calibration
        
        Arguments:
        ----------
            metric_results { dict }  : Dictionary mapping metric names to MetricResult objects

            domain        { Domain } : Text domain for adaptive thresholding
            
        Returns:
        --------
            { EnsembleResult }       : EnsembleResult object with calibrated final prediction
        """
        try:
            # Filter out metrics with errors
            valid_results = self._filter_valid_metrics(results = metric_results)
            
            if (len(valid_results) < self.min_metrics_required):
                logger.warning(f"Insufficient valid metrics: {len(valid_results)}/{self.min_metrics_required}")
                return self._create_fallback_result(domain, metric_results, "insufficient_metrics")
            
            # Get domain-specific base weights
            enabled_metrics                                         = {name: True for name in valid_results.keys()}

            base_weights                                            = get_active_metric_weights(domain          = domain, 
                                                                                                enabled_metrics = enabled_metrics,
                                                                                               )
            
            # Confidence-calibrated aggregation
            aggregated, calculated_weights                          = self._confidence_calibrated_aggregation(results      = valid_results, 
                                                                                                              base_weights = base_weights,
                                                                                                             )
            
            # FIXED: Apply power-based calibration (proper method for probabilities)
            synthetic_prob_cal, authentic_prob_cal, hybrid_prob_cal = self._apply_power_calibration(synthetic_prob = aggregated["synthetic_probability"],
                                                                                                    authentic_prob = aggregated["authentic_probability"],
                                                                                                    hybrid_prob    = aggregated["hybrid_probability"],
                                                                                                    temperature    = self.calibration_temp,
                                                                                                   )
             
            # Assign zero weight to failed metrics
            final_metric_weights                                    = calculated_weights.copy()

            for original_metric_name in metric_results.keys():
                if (original_metric_name not in final_metric_weights):
                    final_metric_weights[original_metric_name] = 0.0

            # Calculate confidence and uncertainty using calibrated probabilities
            calibrated_probabilities                                = {"synthetic_probability" : synthetic_prob_cal,
                                                                       "authentic_probability" : authentic_prob_cal,
                                                                       "hybrid_probability"    : hybrid_prob_cal,
                                                                      }

            overall_confidence                                      = self._calculate_confidence(results    = valid_results, 
                                                                                                 weights    = calculated_weights, 
                                                                                                 aggregated = calibrated_probabilities)

            uncertainty_score                                       = self._calculate_uncertainty(results    = valid_results, 
                                                                                                  aggregated = {"synthetic_probability" : synthetic_prob_cal},
                                                                                                 )

            consensus_level                                         = self._calculate_consensus_level(results = valid_results)
            
            # Apply domain-specific threshold with uncertainty consideration
            domain_thresholds                                       = get_threshold_for_domain(domain = domain)
            
            # Selective abstention
            if ((overall_confidence < params.MIN_CONFIDENCE_FOR_DECISION) or (uncertainty_score > params.MAX_UNCERTAINTY_FOR_DECISION) or (consensus_level < params.MIN_CONSENSUS_FOR_DECISION)):
                
                final_verdict = "Uncertain"
            
            else:
                final_verdict  = self._apply_adaptive_threshold(synthetic_prob = synthetic_prob_cal,
                                                                base_threshold = domain_thresholds.ensemble_threshold,
                                                                uncertainty    = uncertainty_score,
                                                                hybrid_prob    = hybrid_prob_cal,
                                                               )
            
            # Generate reasoning with calibrated probabilities
            reasoning          = self._generate_reasoning(results      = valid_results, 
                                                          weights      = calculated_weights,
                                                          synthetic    = synthetic_prob_cal,
                                                          authentic    = authentic_prob_cal,
                                                          hybrid       = hybrid_prob_cal,
                                                          verdict      = final_verdict, 
                                                          uncertainty  = uncertainty_score, 
                                                          consensus    = consensus_level,
                                                         )
            
            # Calculate weighted scores using calibrated probabilities
            weighted_scores    = {name: (result.synthetic_probability * calculated_weights.get(name, 0.0)) for name, result in valid_results.items()}
            
            return EnsembleResult(final_verdict         = final_verdict,
                                  synthetic_probability = synthetic_prob_cal,
                                  authentic_probability = authentic_prob_cal,
                                  hybrid_probability    = hybrid_prob_cal,
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
    

    def _apply_power_calibration(self, synthetic_prob: float, authentic_prob: float, hybrid_prob: float, temperature: float) -> Tuple[float, float, float]:
        """
        Apply power-based calibration for probability adjustment; uses power transformation
        
        Power transformation is appropriate when working with probabilities directly:
        - T > 1.0: exponent < 1.0 → probabilities move toward 0.5 (softer, less confident)
        - T < 1.0: exponent > 1.0 → probabilities move toward 0 or 1 (sharper, more confident)
        - T = 1.0: exponent = 1.0 → no change
        
        Mathematical background:
        ------------------------
        - For probability p and temperature T: p_calibrated = p^(1/T) / Z; where Z is normalization constant to ensure sum = 1
        
        Arguments:
        ----------
            synthetic_prob { float }   : Original synthetic probability

            authentic_prob { float }   : Original authentic probability
            
            hybrid_prob    { float }   : Original hybrid probability
            
            temperature    { float }   : Calibration temperature (typically 1.0-2.0)
        
        Returns:
        --------
                   { tuple }           : Calibrated (synthetic, authentic, hybrid) probabilities
        """
        # Prevent numerical issues with zero probabilities
        epsilon              = 1e-10  
        
        # Calculate exponent (inverse of temperature): Clamp to prevent division issues
        # T > 1 → exponent < 1 → softer distribution
        # T < 1 → exponent > 1 → sharper distribution
        exponent             = 1.0 / max(temperature, 0.1) 
        
        # Apply power transformation
        synthetic_scaled     = np.power(synthetic_prob + epsilon, exponent)
        authentic_scaled     = np.power(authentic_prob + epsilon, exponent)
        hybrid_scaled        = np.power(hybrid_prob + epsilon, exponent)
        
        # Renormalize to ensure probabilities sum to 1.0
        total                = synthetic_scaled + authentic_scaled + hybrid_scaled
        
        calibrated_synthetic = synthetic_scaled / total
        calibrated_authentic = authentic_scaled / total
        calibrated_hybrid    = hybrid_scaled / total
        
        return calibrated_synthetic, calibrated_authentic, calibrated_hybrid
    

    def _filter_valid_metrics(self, results: Dict[str, MetricResult]) -> Dict[str, MetricResult]:
        """
        Filter out failed metrics (error != None)
        """
        return {name: result for name, result in results.items() if result.error is None}
    

    def _confidence_calibrated_aggregation(self, results: Dict[str, MetricResult], base_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Confidence-calibrated aggregation (single method, simplified)
        """
        # Calculate confidence-adjusted weights
        confidence_weights = {name: base_weights.get(name, 0.0) * self._sigmoid_confidence_adjustment(confidence = result.confidence) for name, result in results.items()}
        
        # Normalize weights
        confidence_weights = self._normalize_weights(weights = confidence_weights)
        
        # Weighted aggregation
        aggregated         = self._weighted_aggregation(results = results, 
                                                        weights = confidence_weights,
                                                       )
        
        return aggregated, confidence_weights
    

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
            return {"synthetic_probability" : params.DEFAULT_SYNTHETIC_PROB, 
                    "authentic_probability" : params.DEFAULT_AUTHENTIC_PROB, 
                    "hybrid_probability"    : params.DEFAULT_HYBRID_PROB,
                   }
        
        # Calculate weighted averages
        synthetic_probability = sum(synthetic_scores) / total_weight
        authentic_probability = sum(authentic_scores) / total_weight
        hybrid_probability    = sum(hybrid_scores) / total_weight
        
        # Normalize probabilities to sum to 1.0
        total_probability     = synthetic_probability + authentic_probability + hybrid_probability

        if (total_probability > 0):
            synthetic_probability /= total_probability
            authentic_probability /= total_probability
            hybrid_probability    /= total_probability
        
        return {"synthetic_probability" : synthetic_probability, 
                "authentic_probability" : authentic_probability,
                "hybrid_probability"    : hybrid_probability,
               }
    

    def _sigmoid_confidence_adjustment(self, confidence: float) -> float:
        """
        Non-linear confidence adjustment using sigmoid
        """
        sigmoid_adjusted_confidence = 1.0 / (1.0 + np.exp(-params.SIGMOID_CONFIDENCE_SCALE * (confidence - params.SIGMOID_CENTER)))
        
        return sigmoid_adjusted_confidence
    

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.0
        """
        total = sum(weights.values())
        
        if total == 0:
            return weights
        
        return {name: weight / total for name, weight in weights.items()}
    

    def _calculate_confidence(self, results: Dict[str, MetricResult], weights: Dict[str, float], aggregated: Dict[str, float]) -> float:
        """
        Confidence calculation
        """
        # Weighted average of metric confidences
        weighted_conf   = sum(result.confidence * weights.get(name, 0.0) for name, result in results.items())
        
        # Agreement factor (lower variance = higher agreement)
        synthetic_probs = [r.synthetic_probability for r in results.values()]
        agreement       = 1.0 - min(1.0, np.std(synthetic_probs) * params.CONSENSUS_STD_SCALING)
        
        # Combined confidence
        confidence      = ((weighted_conf * params.CONFIDENCE_WEIGHT_EVIDENCE) + (agreement * params.CONFIDENCE_WEIGHT_CONSENSUS))
        
        return max(0.0, min(1.0, confidence))
    

    def _calculate_uncertainty(self, results: Dict[str, MetricResult], aggregated: Dict[str, float]) -> float:
        """
        Calculate uncertainty score
        """
        # Variance in predictions
        synthetic_probs        = [r.synthetic_probability for r in results.values()]
        variance_uncertainty   = np.var(synthetic_probs) if (len(synthetic_probs) > 1) else 0.0
        
        # Confidence uncertainty
        avg_confidence         = np.mean([r.confidence for r in results.values()])
        confidence_uncertainty = params.MAX_CONFIDENCE - avg_confidence
        
        # Decision uncertainty (how close to center)
        decision_uncertainty   = params.MAX_DECISION_UNCERTAINTY - params.DECISION_UNCERTAINTY_SCALE * abs(aggregated["synthetic_probability"] - params.DECISION_AMBIGUITY_CENTER)
        
        # Combined uncertainty
        uncertainty            = ((variance_uncertainty * params.UNCERTAINTY_WEIGHT_VARIANCE) + 
                                  (confidence_uncertainty * params.UNCERTAINTY_WEIGHT_CONFIDENCE) + 
                                  (decision_uncertainty * params.UNCERTAINTY_WEIGHT_DECISION)
                                 )
        
        # Finally clip it 
        final_uncertainty      = max(0.0, min(1.0, uncertainty))

        return final_uncertainty
    

    def _calculate_consensus_level(self, results: Dict[str, MetricResult]) -> float:
        """
        Calculate consensus level among metrics
        """
        # Perfect consensus with only one metric
        if (len(results) < 2):
            return 1.0  
        
        synthetic_probabilities = [r.synthetic_probability for r in results.values()]
        std_dev                 = np.std(synthetic_probabilities)
        
        # Convert to consensus level (1.0 = perfect consensus, 0.0 = no consensus)
        consensus               = 1.0 - min(1.0, std_dev * params.CONSENSUS_STD_SCALING)
        
        return consensus
    

    def _apply_adaptive_threshold(self, synthetic_prob: float, base_threshold: float, uncertainty: float, hybrid_prob: float) -> str:
        """
        Apply adaptive threshold considering uncertainty
        """
        # Adjust threshold based on uncertainty (higher uncertainty requires more confidence)
        adjusted_threshold = base_threshold + (uncertainty * params.UNCERTAINTY_THRESHOLD_ADJUSTMENT)
        
        # Check for hybrid content
        if ((hybrid_prob > params.HYBRID_PROB_THRESHOLD) or 
            ((uncertainty > params.HYBRID_UNCERTAINTY_THRESHOLD) and 
             (params.HYBRID_SYNTHETIC_RANGE_LOW <= synthetic_prob <= params.HYBRID_SYNTHETIC_RANGE_HIGH)
            )
           ):
            return "Hybrid"
        
        # Apply threshold with margin
        if (synthetic_prob >= (adjusted_threshold + params.DECISION_MARGIN)):
            return "Synthetically-Generated"
        
        elif (synthetic_prob <= (1.0 - adjusted_threshold - params.DECISION_MARGIN)):
            return "Authentically-Written"
        
        else:
            return "Uncertain"
    

    def _generate_reasoning(self, results: Dict[str, MetricResult], weights: Dict[str, float], synthetic: float, authentic: float, hybrid: float, verdict: str, uncertainty: float, consensus: float) -> List[str]:
        """
        Generate human-readable reasoning for the prediction
        """
        reasoning = list()
        
        # Main verdict explanation
        if (verdict == "Synthetically-Generated-Text"):
            reasoning.append(f"Analysis indicates synthetic-consistency patterns (probability: {synthetic:.2%})")

        elif (verdict == "Authentically-Written-Text"):
            reasoning.append(f"Analysis indicates authentic human-writing patterns (probability: {authentic:.2%})")

        elif (verdict == "Hybrid-Text"):
            reasoning.append(f"Analysis suggests mixed authorship or AI-assisted content (synthetic: {synthetic:.2%}, authentic: {authentic:.2%})")

        else:
            reasoning.append(f"Uncertain classification due to ambiguous patterns (synthetic: {synthetic:.2%}, authentic: {authentic:.2%})")
        
        # Consensus analysis
        if (consensus > params.METRICS_DISAGREEMENT_THRESHOLD_STRONG):
            reasoning.append(f"Strong metric consensus (agreement: {consensus:.2%})")

        elif (consensus > (1.0 - params.METRICS_DISAGREEMENT_THRESHOLD_HIGH)):
            reasoning.append(f"Moderate metric consensus (agreement: {consensus:.2%})")

        else:
            reasoning.append(f"Low metric consensus - conflicting signals (agreement: {consensus:.2%})")
        
        # Uncertainty explanation
        if (uncertainty < 0.3):
            reasoning.append(f"High confidence in assessment (uncertainty: {uncertainty:.2%})")

        elif (uncertainty < 0.5):
            reasoning.append(f"Moderate confidence in assessment (uncertainty: {uncertainty:.2%})")

        else:
            reasoning.append(f"Low confidence - recommend human review (uncertainty: {uncertainty:.2%})")
        
        # Top contributing metrics
        sorted_metrics = sorted(weights.items(), key = lambda x: x[1], reverse = True)
        top_metrics    = [name for name, weight in sorted_metrics[:3] if (weight > 0.01)]
        
        if top_metrics:
            reasoning.append(f"Key signals: {', '.join(top_metrics)}")
        
        return reasoning
    

    def _create_fallback_result(self, domain: Domain, metric_results: Dict[str, MetricResult], error_reason: str) -> EnsembleResult:
        """
        Create fallback result when prediction fails
        """
        return EnsembleResult(final_verdict         = "Uncertain",
                              synthetic_probability = params.DEFAULT_SYNTHETIC_PROB,
                              authentic_probability = params.DEFAULT_AUTHENTIC_PROB,
                              hybrid_probability    = params.DEFAULT_HYBRID_PROB,
                              overall_confidence    = 0.0,
                              domain                = domain,
                              metric_results        = metric_results,
                              metric_weights        = {name: 0.0 for name in metric_results.keys()},
                              weighted_scores       = {name: 0.0 for name in metric_results.keys()},
                              reasoning             = [f"Classification failed: {error_reason}"],
                              uncertainty_score     = 1.0,
                              consensus_level       = 0.0,
                              execution_mode        = self.execution_mode,
                             )


# Export
__all__ = ["EnsembleClassifier"]