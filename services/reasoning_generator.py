# DEPENDENCIES
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from config.enums import Domain
from config.schemas import MetricResult
from config.schemas import EnsembleResult
from config.schemas import DetailedReasoningResult


class ReasoningGenerator:
    """
    Generates detailed, human-readable reasoning for Synthetic detection results with ensemble and domain-aware integration
    
    Features:
    - Ensemble method explanation
    - Domain-aware calibration context
    - Uncertainty quantification
    - Metric contribution analysis
    - Actionable recommendations
    """
    # Metric descriptions 
    METRIC_DESCRIPTIONS  = {"structural"                   : "analyzes sentence structure, length patterns, and statistical features",
                            "perplexity"                   : "measures text predictability using language model cross-entropy",
                            "entropy"                      : "evaluates token diversity and sequence unpredictability",
                            "semantic_analysis"            : "examines semantic coherence, topic consistency, and logical flow",
                            "linguistic"                   : "assesses grammatical patterns, syntactic complexity, and style markers",
                            "multi_perturbation_stability" : "tests text stability under perturbation using curvature analysis",
                           }
    
    # Ensemble method descriptions
    ENSEMBLE_METHODS     = {"confidence_calibrated" : "confidence-weighted aggregation with domain calibration",
                            "consensus_based"       : "rewarding metric agreement and consensus",
                            "domain_weighted"       : "domain-aware static weighting of metrics",
                            "simple_average"        : "equal weighting of all metrics",
                           }
    
    # Synthetic indicators aligned with current metric outputs
    SYNTHETIC_INDICATORS = {"low_perplexity"          : "Text shows high predictability to language models",
                            "low_entropy"             : "Limited vocabulary diversity and repetitive patterns",
                            "structural_uniformity"   : "Consistent sentence lengths and structural patterns",
                            "semantic_perfection"     : "Unnaturally perfect coherence and logical flow",
                            "linguistic_consistency"  : "Overly consistent grammatical patterns and style",
                            "perturbation_instability": "Text changes significantly under minor modifications",
                            "low_burstiness"          : "Lacks natural variation in writing intensity",
                            "transition_overuse"      : "Excessive use of transitional phrases and connectors",
                           }
    
    # Authentic indicators
    AUTHENTIC_INDICATORS = {"high_perplexity"       : "Creative, unpredictable word choices and phrasing",
                            "high_entropy"          : "Rich vocabulary diversity and varied expressions",
                            "structural_variation"  : "Natural variation in sentence lengths and structures",
                            "semantic_naturalness"  : "Authentic, occasionally imperfect logical flow",
                            "linguistic_diversity"  : "Varied grammatical constructions and personal style",
                            "perturbation_stability": "Text remains consistent under minor modifications",
                            "high_burstiness"       : "Natural variation in writing intensity and focus",
                            "personal_voice"        : "Distinctive personal expressions and idioms",
                           }

    
    def __init__(self):
        """
        Initialize reasoning generator with ensemble awareness
        """
        pass
    

    def generate(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult], domain: Domain, text_length: int = 0, ensemble_method: str = "confidence_calibrated") -> DetailedReasoningResult:
        """
        Generate comprehensive reasoning for detection result with ensemble integration
        
        Arguments:
        ----------
            ensemble_result   { EnsembleResult}  : Final ensemble prediction with weights and reasoning

            metric_results         { dict }      : Individual metric results from all metrics
            
            domain                { Domain }     : Detected text domain for context-aware analysis
            
            text_length            { int }       : Length of analyzed text in words
            
            ensemble_method        { str }       : Method used for ensemble aggregation
            
        Returns:
        --------
               { DetailedReasoningResult }       : DetailedReasoningResult object with ensemble-aware analysis
        """
        # Generate summary with ensemble context
        summary                                     = self._generate_ensemble_summary(ensemble_result = ensemble_result, 
                                                                                      domain          = domain, 
                                                                                      text_length     = text_length, 
                                                                                      ensemble_method = ensemble_method,
                                                                                     )
        
        # Identify key indicators with metric weights
        key_indicators                              = self._identify_weighted_indicators(ensemble_result = ensemble_result, 
                                                                                         metric_results  = metric_results,
                                                                                        )
        
        # Generate metric explanations with confidence
        metric_explanations                         = self._generate_metric_explanations(metric_results = metric_results, 
                                                                                         metric_weights = ensemble_result.metric_weights,
                                                                                        )
        
        # Compile evidence with ensemble consensus
        supporting_evidence, contradicting_evidence = self._compile_ensemble_evidence(ensemble_result = ensemble_result, 
                                                                                      metric_results  = metric_results,
                                                                                     )
        
        # Explain confidence with uncertainty
        confidence_explanation                      = self._explain_confidence_with_uncertainty(ensemble_result = ensemble_result, 
                                                                                                metric_results  = metric_results,
                                                                                               )
        
        # Domain-specific analysis
        domain_analysis                             = self._generate_domain_analysis(domain          = domain, 
                                                                                     metric_results  = metric_results, 
                                                                                     ensemble_result = ensemble_result,
                                                                                    )
        
        # Ensemble methodology explanation
        ensemble_analysis                           = self._explain_ensemble_methodology(ensemble_result = ensemble_result, 
                                                                                         ensemble_method = ensemble_method,
                                                                                        )
        
        # Uncertainty analysis
        uncertainty_analysis                        = self._analyze_uncertainty(ensemble_result = ensemble_result)
        
        # Generate recommendations
        recommendations                             = self._generate_ensemble_recommendations(ensemble_result = ensemble_result,
                                                                                              metric_results  = metric_results, 
                                                                                              domain          = domain,
                                                                                             )
        
        return DetailedReasoningResult(summary                = summary,
                                       key_indicators         = key_indicators,
                                       metric_explanations    = metric_explanations,
                                       supporting_evidence    = supporting_evidence,
                                       contradicting_evidence = contradicting_evidence,
                                       confidence_explanation = confidence_explanation,
                                       domain_analysis        = domain_analysis,
                                       ensemble_analysis      = ensemble_analysis,
                                       recommendations        = recommendations,
                                       uncertainty_analysis   = uncertainty_analysis,
                                      )
    

    def _generate_ensemble_summary(self, ensemble_result: EnsembleResult, domain: Domain, text_length: int, ensemble_method: str) -> str:
        """
        Generate executive summary with ensemble context
        """
        verdict        = ensemble_result.final_verdict
        synthetic_prob = ensemble_result.synthetic_probability
        authentic_prob = ensemble_result.authentic_probability
        hybrid_prob    = ensemble_result.hybrid_probability
        confidence     = ensemble_result.overall_confidence
        uncertainty    = ensemble_result.uncertainty_score
        consensus      = ensemble_result.consensus_level
        
        # Confidence level description
        if (confidence >= 0.8):
            conf_desc = "very high confidence"

        elif (confidence >= 0.6):
            conf_desc = "high confidence"

        elif (confidence >= 0.4):
            conf_desc = "moderate confidence"

        else:
            conf_desc = "low confidence"
        
        # Consensus description
        if (consensus >= 0.8):
            consensus_desc = "strong consensus"

        elif (consensus >= 0.6):
            consensus_desc = "moderate consensus"

        else:
            consensus_desc = "low consensus"
        
        # Build summary based on verdict
        summary_parts = list()
        
        if (verdict == "Synthetically-Generated"):
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text is "
                                 f"**likely synthetically-generated** (synthetic probability: {synthetic_prob:.1%}).")
        
        elif( verdict == "Authentically-Written"):
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text is "
                                 f"**likely authentically-written** (authentic probability: {authentic_prob:.1%}).")
        
        elif (verdict == "Hybrid"):
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text "
                                 f"**contains mixed synthetic/authentic content** (hybrid probability: {hybrid_prob:.1%}).")
        
        else:  
            # Uncertain
            summary_parts.append(f"Ensemble analysis is **inconclusive** (confidence: {confidence:.1%}).")
        
        # Add ensemble context
        summary_parts.append(f"Metrics show {consensus_desc} among detection methods. Uncertainty level: {uncertainty:.1%}.")
        
        # Add domain and length context
        if (text_length > 0):
            summary_parts.append(f"Analysis of {text_length:,} words in **{domain.value}** domain using {self.ENSEMBLE_METHODS.get(ensemble_method, ensemble_method)} ensemble method.")
        
        else:
            summary_parts.append(f"Analysis in **{domain.value}** domain using {self.ENSEMBLE_METHODS.get(ensemble_method, ensemble_method)} ensemble method.")
        
        return " ".join(summary_parts)
    

    def _identify_weighted_indicators(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult]) -> List[str]:
        """
        Identify top indicators considering metric weights and contributions
        """
        indicators       = list()
        is_synthetic     = (ensemble_result.final_verdict == "Synthetically-Generated")
        
        # Use ensemble weights to prioritize indicators
        weighted_metrics = []
        
        for name, result in metric_results.items():
            if result.error:
                continue
            
            weight         = ensemble_result.metric_weights.get(name, 0.0)
            confidence     = result.confidence
            # Combine weight and confidence for prioritization
            priority_score = weight * confidence
            
            weighted_metrics.append((name, result, priority_score))
        
        # Sort by priority score
        weighted_metrics.sort(key = lambda x: x[2], reverse = True)
        
        for name, result, priority_score in weighted_metrics[:5]:  
            # Top 5 metrics
            key_feature = self._extract_ensemble_feature(name, result, is_synthetic, priority_score)
            
            if key_feature:
                weight_pct = ensemble_result.metric_weights.get(name, 0.0) * 100
                indicators.append(f"**{name.title()}** ({weight_pct:.1f}% weight): {key_feature}")
        
        return indicators
    

    def _extract_ensemble_feature(self, metric_name: str, result: MetricResult, is_synthetic: bool, priority_score: float) -> Optional[str]:
        """
        Extract significant features considering ensemble context
        """
        details = result.details
        
        if (metric_name == "structural"):
            burstiness = details.get("burstiness_score", 0.5)
            uniformity = details.get("length_uniformity", 0.5)
            
            if (is_synthetic and (burstiness < 0.4)):
                return f"Low burstiness ({burstiness:.2f}) suggests uniform synthetic patterns"
            
            elif (not is_synthetic and (burstiness > 0.6)):
                return f"High burstiness ({burstiness:.2f}) indicates natural variation"
            
            elif (is_synthetic and (uniformity > 0.7)):
                return f"High structural uniformity ({uniformity:.2f}) typical of synthetic text"
            
        elif (metric_name == "perplexity"):
            perplexity = details.get("overall_perplexity", 50)
            
            if (is_synthetic and perplexity < 35):
                return f"Low perplexity ({perplexity:.1f}) indicates high predictability"
            
            elif (not is_synthetic and (perplexity > 55)):
                return f"High perplexity ({perplexity:.1f}) suggests human creativity"
            
        elif (metric_name == "entropy"):
            token_diversity = details.get("token_diversity", 0.5)
            
            if (is_synthetic and (token_diversity < 0.65)):
                return f"Low token diversity ({token_diversity:.2f}) suggests synthetic patterns"
            
            elif (not is_synthetic and (token_diversity > 0.75)):
                return f"High token diversity ({token_diversity:.2f}) indicates human variety"
            
        elif (metric_name == "semantic_analysis"):
            coherence = details.get("coherence_score", 0.5)
            
            if (is_synthetic and (coherence > 0.8)):
                return f"Unnaturally high coherence ({coherence:.2f}) typical of synthetic text"
            
            elif (not is_synthetic and (0.4 <= coherence <= 0.7)):
                return f"Natural coherence variation ({coherence:.2f})"
            
        elif (metric_name == "linguistic"):
            pos_diversity = details.get("pos_diversity", 0.5)
            
            if (is_synthetic and (pos_diversity < 0.4)):
                return f"Limited grammatical diversity ({pos_diversity:.2f})"
            
            elif (not is_synthetic and (pos_diversity > 0.55)):
                return f"Rich grammatical variety ({pos_diversity:.2f})"
                
        elif (metric_name == "multi_perturbation_stability"):
            stability = details.get("stability_score", 0.5)
            
            if (is_synthetic and (stability > 0.6)):
                return f"High perturbation sensitivity ({stability:.2f})"
            
            elif (not is_synthetic and (stability < 0.4)):
                return f"Text stability under perturbation ({stability:.2f})"
        
        return None
    

    def _generate_metric_explanations(self, metric_results: Dict[str, MetricResult], metric_weights: Dict[str, float]) -> Dict[str, str]:
        """
        Generate explanations for each metric with weight context
        """
        explanations = dict()
        
        for name, result in metric_results.items():
            if result.error:
                explanations[name] = f"⚠️ Analysis failed: {result.error}"
                continue
            
            # Get metric description
            desc        = self.METRIC_DESCRIPTIONS.get(name, "analyzes text characteristics")
            
            # Get weight information
            weight      = metric_weights.get(name, 0.0)
            weight_info = f" (ensemble weight: {weight:.1%})" if weight > 0 else " (low weight in ensemble)"
            
            # Determine verdict based on probabilities
            if (result.synthetic_probability > 0.6):
                verdict = "suggests synthetic generation"
                prob    = result.synthetic_probability
            
            elif (result.authentic_probability > 0.6):
                verdict = "indicates authentic writing"
                prob    = result.authentic_probability
            
            else:
                verdict = "shows mixed signals"
                prob    = max(result.synthetic_probability, result.authentic_probability)
            
            # Build explanation with confidence
            explanation = (f"This metric {desc}.{weight_info} Result: {verdict} ({prob:.1%} probability) with {result.confidence:.1%} confidence.")
            
            explanations[name] = explanation
        
        return explanations
    

    def _compile_ensemble_evidence(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult]) -> tuple:
        """
        Compile evidence considering ensemble consensus and weights
        """
        is_synthetic_verdict = (ensemble_result.final_verdict == "Synthetically-Generated")
        consensus            = ensemble_result.consensus_level
        
        supporting           = list()
        contradicting        = list()
        
        for name, result in metric_results.items():
            if result.error:
                continue
            
            weight                    = ensemble_result.metric_weights.get(name, 0.0)
            metric_suggests_synthetic = (result.synthetic_probability > result.authentic_probability)
            
            # Weight the evidence by metric importance
            weight_indicator          = "🟢" if (weight > 0.15) else "🟡" if (weight > 0.08) else "⚪"
            
            if (metric_suggests_synthetic == is_synthetic_verdict):
                # Supporting evidence
                indicator = self._get_synthetic_indicator_from_metric(name, result) if is_synthetic_verdict else self._get_authentic_indicator_from_metric(name, result)
                
                if indicator:
                    supporting.append(f"{weight_indicator} {indicator}")
            
            else:
                # Contradicting evidence
                indicator = self._get_authentic_indicator_from_metric(name, result) if is_synthetic_verdict else self._get_synthetic_indicator_from_metric(name, result)
                
                if indicator:
                    contradicting.append(f"{weight_indicator} {indicator}")
        
        # Add consensus context
        if (consensus > 0.7):
            supporting.insert(0, "✅ Strong metric consensus supports this conclusion")
        
        elif (consensus < 0.4):
            contradicting.insert(0, "⚠️ Low metric consensus indicates uncertainty")
        
        return supporting, contradicting
    

    def _get_synthetic_indicator_from_metric(self, metric_name: str, result: MetricResult) -> Optional[str]:
        """
        Get synthetic indicator from metric result
        """
        details = result.details
        
        if (metric_name == "structural"):
            if (details.get("burstiness_score", 1.0) < 0.4):
                return self.SYNTHETIC_INDICATORS["low_burstiness"]
            
        elif (metric_name == "perplexity"):
            if (details.get("overall_perplexity", 100) < 35):
                return self.SYNTHETIC_INDICATORS["low_perplexity"]
            
        elif (metric_name == "entropy"):
            if (details.get("token_diversity", 1.0) < 0.65):
                return self.SYNTHETIC_INDICATORS["low_entropy"]
            
        elif (metric_name == "semantic_analysis"):
            if (details.get("coherence_score", 0.5) > 0.75):
                return self.SYNTHETIC_INDICATORS["semantic_perfection"]
            
        return None
    

    def _get_authentic_indicator_from_metric(self, metric_name: str, result: MetricResult) -> Optional[str]:
        """
        Get authentic indicator from metric result
        """
        details = result.details
        
        if (metric_name == "structural"):
            if (details.get("burstiness_score", 0.0) > 0.6):
                return self.AUTHENTIC_INDICATORS["high_burstiness"]
            
        elif (metric_name == "perplexity"):
            if (details.get("overall_perplexity", 0) > 55):
                return self.AUTHENTIC_INDICATORS["high_perplexity"]
            
        elif (metric_name == "entropy"):
            if (details.get("token_diversity", 0.0) > 0.75):
                return self.AUTHENTIC_INDICATORS["high_entropy"]
            
        return None
    

    def _explain_confidence_with_uncertainty(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult]) -> str:
        """
        Explain confidence considering uncertainty metrics
        """
        confidence        = ensemble_result.overall_confidence
        uncertainty       = ensemble_result.uncertainty_score
        consensus         = ensemble_result.consensus_level
        
        # Calculate additional factors
        valid_metrics     = len([r for r in metric_results.values() if not r.error])
        high_conf_metrics = len([r for r in metric_results.values() if not r.error and r.confidence > 0.7])
        
        explanation       = f"**Confidence: {confidence:.1%}** | **Uncertainty: {uncertainty:.1%}** | **Consensus: {consensus:.1%}**\n\n"
        
        if (confidence >= 0.8):
            explanation += "High confidence due to: strong metric agreement, clear patterns, and reliable signal across multiple detection methods."
        
        elif (confidence >= 0.6):
            explanation += "Good confidence supported by: general metric agreement and consistent detection patterns."
        
        else:
            explanation += "Lower confidence reflects: metric disagreement, ambiguous patterns, or borderline characteristics."
        
        explanation += f"\n\n• {high_conf_metrics}/{valid_metrics} metrics with high confidence"
        explanation += f"\n• Ensemble uncertainty score: {uncertainty:.1%}"
        explanation += f"\n• Metric consensus level: {consensus:.1%}"
        
        return explanation
    

    def _generate_domain_analysis(self, domain: Domain, metric_results: Dict[str, MetricResult], ensemble_result: EnsembleResult) -> str:
        """
        Generate domain-specific analysis with calibration context
        """
        domain_contexts = {Domain.ACADEMIC      : "Academic writing analysis emphasizes: citation patterns, technical depth, argument structure, and formal tone. Detection calibrated for scholarly conventions.",
                           Domain.CREATIVE      : "Creative writing analysis focuses: narrative voice, emotional authenticity, stylistic variation, and imaginative elements. Accounts for artistic license.",
                           Domain.TECHNICAL_DOC : "Technical documentation analysis examines: specialized terminology, structured explanations, practical examples, and precision requirements.",
                           Domain.SOCIAL_MEDIA  : "Social media analysis considers: informal language, brevity, emotional expression, and platform-specific conventions.",
                           Domain.GENERAL       : "General content analysis uses universal patterns across writing styles and genres.",
                          }
        
        context         = domain_contexts.get(domain, domain_contexts[Domain.GENERAL])
        
        # Add domain-specific threshold context
        threshold_info  = {Domain.ACADEMIC      : "Higher detection thresholds applied for academic rigor",
                           Domain.TECHNICAL_DOC : "Elevated thresholds for technical precision requirements",
                           Domain.CREATIVE      : "Balanced thresholds accounting for creative expression",
                           Domain.SOCIAL_MEDIA  : "Adapted thresholds for informal communication patterns",
                           Domain.GENERAL       : "Standard detection thresholds applied",
                          }
        
        threshold_note  = threshold_info.get(domain, "Standard detection thresholds applied")
        
        return f"**Domain Analysis ({domain.value})**\n\n{context}\n\n{threshold_note}"
    

    def _explain_ensemble_methodology(self, ensemble_result: EnsembleResult, ensemble_method: str) -> str:
        """
        Explain the ensemble methodology used
        """
        method_desc = self.ENSEMBLE_METHODS.get(ensemble_method, "advanced aggregation of multiple detection methods")
        
        explanation = f"**Ensemble Methodology**: {method_desc}\n\n"
        
        # Explain key top metrics
        top_metrics = sorted(ensemble_result.metric_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_metrics:
            explanation += "**Top contributing metrics**:\n"
            for metric, weight in top_metrics:
                explanation += f"• {metric}: {weight:.1%} weight\n"
        
        # Add reasoning snippets if available
        if hasattr(ensemble_result, 'reasoning') and ensemble_result.reasoning:
            # Filter out section headers and take first 2 key reasons
            key_reasons = [r for r in ensemble_result.reasoning if not r.startswith('##')][:2]
            if key_reasons:
                explanation += "\n**Key ensemble factors**:\n"
                for reason in key_reasons:
                    # Clean up the reason text
                    clean_reason = reason.replace('**', '').replace('✓', '').replace('⚠', '').strip()
                    explanation += f"• {clean_reason}\n"
        
        return explanation
    

    def _analyze_uncertainty(self, ensemble_result: EnsembleResult) -> str:
        """
        Analyze and explain uncertainty factors
        """
        uncertainty = ensemble_result.uncertainty_score
        
        if (uncertainty < 0.3):
            return "**Low Uncertainty**: Clear detection signals with strong metric agreement. Results are highly reliable."
        
        elif (uncertainty < 0.6):
            return "**Moderate Uncertainty**: Some metric disagreement or borderline characteristics. Consider additional context."
        
        else:
            return "**High Uncertainty**: Significant metric disagreement or ambiguous patterns. Results should be interpreted with caution and additional verification may be needed."
    

    def _generate_ensemble_recommendations(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult], domain: Domain) -> List[str]:
        """
        Generate actionable recommendations based on ensemble results
        """
        recommendations = list()
        verdict         = ensemble_result.final_verdict
        confidence      = ensemble_result.overall_confidence
        uncertainty     = ensemble_result.uncertainty_score
        
        # Base recommendations by verdict and confidence
        if (verdict == "Synthetically-Generated"):
            if (confidence >= 0.8):
                recommendations.append("**High-confidence synthetic detection**: Consider verified original drafts or alternative assessment methods.")
            
            else:
                recommendations.append("**Likely synthetic involvement**: Recommend discussion about AI tool usage and verification of understanding.")
        
        elif (verdict == "Authentically-Written"):
            if (confidence >= 0.8):
                recommendations.append("**High-confidence authentic authorship**: No additional verification typically needed.")
            
            else:
                recommendations.append("**Likely authentically-written**: Consider context and writing history for complete assessment.")
        
        elif (verdict == "Hybrid"):
            recommendations.append("**Mixed synthetic/authentic content**: Common in collaborative writing. Discuss appropriate AI use guidelines.")
        
        elif (verdict == "Uncertain"):
            recommendations.append("**Inconclusive result**: The analysis could not reach a clear determination. Additional context or verification may be needed.")
        
        # Uncertainty-based recommendations
        if (uncertainty > 0.6):
            recommendations.append("**High uncertainty case**: Consider complementary verification methods like oral discussion or process documentation.")
        
        # Domain-specific recommendations
        domain_recs = {Domain.ACADEMIC      : "For academic work: verify subject mastery through targeted questions or practical application.",
                       Domain.CREATIVE      : "For creative work: assess originality, personal voice, and creative process documentation.",
                       Domain.TECHNICAL_DOC : "For technical content: verify practical expertise and problem-solving ability.",
                       Domain.SOFTWARE_DEV  : "For code documentation: verify understanding through code review or implementation questions.",
                      }
        
        if domain in domain_recs:
            recommendations.append(domain_recs[domain])
        
        # General best practices
        recommendations.extend(["**Context matters**: Consider author's background, writing history, and situational factors.",
                                "**Educational approach**: Use detection results as conversation starters about appropriate AI use.",
                                "**Continuous evaluation**: AI writing evolves rapidly; regular calibration updates maintain accuracy."
                              ])
        
        return recommendations



# Export
__all__ = ["ReasoningGenerator"]