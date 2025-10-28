# DEPENDENCIES
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import dataclass
from detector.attribution import AIModel
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from detector.ensemble import EnsembleResult
from detector.attribution import AttributionResult 



@dataclass
class DetailedReasoning:
    """
    Comprehensive reasoning for detection result with ensemble integration
    """
    summary                : str
    key_indicators         : List[str]
    metric_explanations    : Dict[str, str]
    supporting_evidence    : List[str]
    contradicting_evidence : List[str]
    confidence_explanation : str
    domain_analysis        : str
    ensemble_analysis      : str
    attribution_reasoning  : Optional[str]
    recommendations        : List[str]
    uncertainty_analysis   : str
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"summary"                : self.summary,
                "key_indicators"         : self.key_indicators,
                "metric_explanations"    : self.metric_explanations,
                "supporting_evidence"    : self.supporting_evidence,
                "contradicting_evidence" : self.contradicting_evidence,
                "confidence_explanation" : self.confidence_explanation,
                "domain_analysis"        : self.domain_analysis,
                "ensemble_analysis"      : self.ensemble_analysis,
                "attribution_reasoning"  : self.attribution_reasoning,
                "recommendations"        : self.recommendations,
                "uncertainty_analysis"   : self.uncertainty_analysis,
               }



class ReasoningGenerator:
    """
    Generates detailed, human-readable reasoning for AI detection results with ensemble and domain-aware integration
    
    Features:
    - Ensemble method explanation
    - Domain-aware calibration context
    - Uncertainty quantification
    - Metric contribution analysis
    - Actionable recommendations
    """
    # Enhanced metric descriptions aligned with current architecture
    METRIC_DESCRIPTIONS = {"structural"        : "analyzes sentence structure, length patterns, and statistical features",
                           "perplexity"        : "measures text predictability using language model cross-entropy",
                           "entropy"           : "evaluates token diversity and sequence unpredictability",
                           "semantic_analysis" : "examines semantic coherence, topic consistency, and logical flow",
                           "linguistic"        : "assesses grammatical patterns, syntactic complexity, and style markers",
                           "detect_gpt"        : "tests text stability under perturbation using curvature analysis",
                          }
    
    # Ensemble method descriptions
    ENSEMBLE_METHODS    = {"confidence_calibrated" : "confidence-weighted aggregation with domain calibration",
                           "domain_adaptive"       : "domain-specific metric performance weighting",
                           "consensus_based"       : "rewarding metric agreement and consensus",
                           "ml_ensemble"           : "machine learning-based meta-classification",
                           "domain_weighted"       : "domain-aware static weighting of metrics",
                          }
    
    # AI indicators aligned with current metric outputs
    AI_INDICATORS       = {"low_perplexity"          : "Text shows high predictability to language models",
                           "low_entropy"             : "Limited vocabulary diversity and repetitive patterns",
                           "structural_uniformity"   : "Consistent sentence lengths and structural patterns",
                           "semantic_perfection"     : "Unnaturally perfect coherence and logical flow",
                           "linguistic_consistency"  : "Overly consistent grammatical patterns and style",
                           "perturbation_instability": "Text changes significantly under minor modifications",
                           "low_burstiness"          : "Lacks natural variation in writing intensity",
                           "transition_overuse"      : "Excessive use of transitional phrases and connectors",
                          }
    
    # Human indicators
    HUMAN_INDICATORS    = {"high_perplexity"       : "Creative, unpredictable word choices and phrasing",
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
    

    def generate(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult], domain: Domain, attribution_result: Optional[AttributionResult] = None, 
                 text_length: int = 0, ensemble_method: str = "confidence_calibrated") -> DetailedReasoning:
        """
        Generate comprehensive reasoning for detection result with ensemble integration
        
        Arguments:
        ----------
            ensemble_result    : Final ensemble prediction with weights and reasoning

            metric_results     : Individual metric results from all 6 metrics
            
            domain             : Detected text domain for context-aware analysis
            
            attribution_result : Model attribution (if available)
            
            text_length        : Length of analyzed text in words
            
            ensemble_method    : Method used for ensemble aggregation
            
        Returns:
        --------
            DetailedReasoning object with ensemble-aware analysis
        """
        # Generate summary with ensemble context
        summary                                     = self._generate_ensemble_summary(ensemble_result, domain, text_length, ensemble_method)
        
        # Identify key indicators with metric weights
        key_indicators                              = self._identify_weighted_indicators(ensemble_result, metric_results)
        
        # Generate metric explanations with confidence
        metric_explanations                         = self._generate_metric_explanations(metric_results, ensemble_result.metric_weights)
        
        # Compile evidence with ensemble consensus
        supporting_evidence, contradicting_evidence = self._compile_ensemble_evidence(ensemble_result, metric_results)
        
        # Explain confidence with uncertainty
        confidence_explanation                      = self._explain_confidence_with_uncertainty(ensemble_result, metric_results)
        
        # Domain-specific analysis
        domain_analysis                             = self._generate_domain_analysis(domain, metric_results, ensemble_result)
        
        # Ensemble methodology explanation
        ensemble_analysis                           = self._explain_ensemble_methodology(ensemble_result, ensemble_method)
        
        # Attribution reasoning
        attribution_reasoning                       = None

        if attribution_result:
            attribution_reasoning = self._generate_attribution_reasoning(attribution_result)
        
        # Uncertainty analysis
        uncertainty_analysis = self._analyze_uncertainty(ensemble_result)
        
        # Generate recommendations
        recommendations      = self._generate_ensemble_recommendations(ensemble_result, metric_results, domain)
        
        return DetailedReasoning(summary                = summary,
                                 key_indicators         = key_indicators,
                                 metric_explanations    = metric_explanations,
                                 supporting_evidence    = supporting_evidence,
                                 contradicting_evidence = contradicting_evidence,
                                 confidence_explanation = confidence_explanation,
                                 domain_analysis        = domain_analysis,
                                 ensemble_analysis      = ensemble_analysis,
                                 attribution_reasoning  = attribution_reasoning,
                                 recommendations        = recommendations,
                                 uncertainty_analysis   = uncertainty_analysis,
                                )
    

    def _generate_ensemble_summary(self, ensemble_result: EnsembleResult, domain: Domain, text_length: int, ensemble_method: str) -> str:
        """
        Generate executive summary with ensemble context
        """
        verdict     = ensemble_result.final_verdict
        ai_prob     = ensemble_result.ai_probability
        confidence  = ensemble_result.overall_confidence
        uncertainty = ensemble_result.uncertainty_score
        consensus   = ensemble_result.consensus_level
        
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
        
        # Build summary based on verdict and ensemble metrics
        summary_parts = list()
        
        if ("AI-Generated" in verdict):
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text is "
                                 f"**likely AI-generated** (AI probability: {ai_prob:.1%})."
                                )

        elif ("Human-Written" in verdict):
            human_prob = ensemble_result.human_probability
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text is "
                                 f"**likely human-written** (human probability: {human_prob:.1%})."
                                )

        elif( "Mixed" in verdict):
            mixed_prob = ensemble_result.mixed_probability
            summary_parts.append(f"Ensemble analysis indicates with {conf_desc} ({confidence:.1%}) that this text "
                                 f"**contains mixed AI-human content** (mixed probability: {mixed_prob:.1%})."
                                )
        
        else:
            summary_parts.append(f"Ensemble analysis is **inconclusive** (confidence: {confidence:.1%}).")
        
        # Add ensemble context
        summary_parts.append(f"Metrics show {consensus_desc} among detection methods. Uncertainty level: {uncertainty:.1%}.")
        
        # Add domain and length context
        summary_parts.append(f"Analysis of {text_length:,} words in **{domain.value}** domain using {self.ENSEMBLE_METHODS.get(ensemble_method, ensemble_method)} ensemble method.")
        
        return " ".join(summary_parts)
    

    def _identify_weighted_indicators(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult]) -> List[str]:
        """
        Identify top indicators considering metric weights and contributions
        """
        indicators       = list()
        is_ai            = "AI-Generated" in ensemble_result.final_verdict
        
        # Use ensemble weights to prioritize indicators
        weighted_metrics = list()

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
            key_feature = self._extract_ensemble_feature(name, result, is_ai, priority_score)
            
            if key_feature:
                weight_pct = ensemble_result.metric_weights.get(name, 0.0) * 100
                indicators.append(f"**{name.title()}** ({weight_pct:.1f}% weight): {key_feature}")
        
        return indicators
    

    def _extract_ensemble_feature(self, metric_name: str, result: MetricResult, is_ai: bool, priority_score: float) -> Optional[str]:
        """
        Extract significant features considering ensemble context
        """
        details = result.details
        
        if (metric_name == "structural"):
            burstiness = details.get("burstiness_score", 0.5)
            uniformity = details.get("length_uniformity", 0.5)
            
            if (is_ai and (burstiness < 0.4)):
                return f"Low burstiness ({burstiness:.2f}) suggests uniform AI patterns"
            
            elif (not is_ai and (burstiness > 0.6)):
                return f"High burstiness ({burstiness:.2f}) indicates natural variation"

            elif (is_ai and (uniformity > 0.7)):
                return f"High structural uniformity ({uniformity:.2f}) typical of AI"
                

        elif (metric_name == "perplexity"):
            perplexity = details.get("overall_perplexity", 50)
            
            if (is_ai and (perplexity < 35)):
                return f"Low perplexity ({perplexity:.1f}) indicates high predictability"

            elif (not is_ai and (perplexity > 55)):
                return f"High perplexity ({perplexity:.1f}) suggests human creativity"
                

        elif (metric_name == "entropy"):
            token_diversity  = details.get("token_diversity", 0.5)
            sequence_entropy = details.get("sequence_entropy", 0.5)
            
            if (is_ai and (token_diversity < 0.65)):
                return f"Low token diversity ({token_diversity:.2f}) suggests AI patterns"

            elif (not is_ai and (token_diversity > 0.75)):
                return f"High token diversity ({token_diversity:.2f}) indicates human variety"


        elif (metric_name == "semantic_analysis"):
            coherence   = details.get("coherence_score", 0.5)
            consistency = details.get("consistency_score", 0.5)
            
            if (is_ai and (coherence > 0.8)):
                return f"Unnaturally high coherence ({coherence:.2f}) typical of AI"

            elif (not is_ai and (0.4 <= coherence <= 0.7)):
                return f"Natural coherence variation ({coherence:.2f})"
                

        elif (metric_name == "linguistic"):
            pos_diversity        = details.get("pos_diversity", 0.5)
            syntactic_complexity = details.get("syntactic_complexity", 2.5)
            
            if (is_ai and (pos_diversity < 0.4)):
                return f"Limited grammatical diversity ({pos_diversity:.2f})"

            elif (not is_ai and (pos_diversity > 0.55)):
                return f"Rich grammatical variety ({pos_diversity:.2f})"
                
        elif (metric_name == "detect_gpt"):
            stability = details.get("stability_score", 0.5)
            curvature = details.get("curvature_score", 0.5)
            
            if (is_ai and (stability > 0.6)):
                return f"High perturbation instability ({stability:.2f})"

            elif (not is_ai and (stability < 0.4)):
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
            desc         = self.METRIC_DESCRIPTIONS.get(name, "analyzes text characteristics")
            
            # Get weight information
            weight      = metric_weights.get(name, 0.0)
            weight_info = f" (ensemble weight: {weight:.1%})" if weight > 0 else " (low weight in ensemble)"
            
            # Determine verdict
            if (result.ai_probability > 0.6):
                verdict = "suggests AI generation"
                prob    = result.ai_probability

            elif (result.human_probability > 0.6):
                verdict = "indicates human writing"
                prob    = result.human_probability

            else:
                verdict = "shows mixed signals"
                prob    = max(result.ai_probability, result.human_probability)
            
            # Build explanation with confidence
            explanation        = (f"This metric {desc}.{weight_info} "
                                  f"Result: {verdict} ({prob:.1%} probability) "
                                  f"with {result.confidence:.1%} confidence."
                                 )
            
            explanations[name] = explanation
        
        return explanations
    

    def _compile_ensemble_evidence(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult]) -> tuple:
        """
        Compile evidence considering ensemble consensus and weights
        """
        is_ai_verdict = "AI-Generated" in ensemble_result.final_verdict
        consensus     = ensemble_result.consensus_level
        
        supporting    = list()
        contradicting = list()
        
        for name, result in metric_results.items():
            if result.error:
                continue
            
            weight             = ensemble_result.metric_weights.get(name, 0.0)
            metric_suggests_ai = result.ai_probability > result.human_probability
            
            # Weight the evidence by metric importance
            weight_indicator   = "🟢" if weight > 0.15 else "🟡" if weight > 0.08 else "⚪"
            
            if (metric_suggests_ai == is_ai_verdict):
                # Supporting evidence
                indicator = self._get_ai_indicator_from_metric(name, result) if is_ai_verdict else self._get_human_indicator_from_metric(name, result)
                
                if indicator:
                    supporting.append(f"{weight_indicator} {indicator}")
            
            else:
                # Contradicting evidence
                indicator = self._get_human_indicator_from_metric(name, result) if is_ai_verdict else self._get_ai_indicator_from_metric(name, result)
                
                if indicator:
                    contradicting.append(f"{weight_indicator} {indicator}")
        
        # Add consensus context
        if (consensus > 0.7):
            supporting.insert(0, "✅ Strong metric consensus supports this conclusion")

        elif (consensus < 0.4):
            contradicting.insert(0, "⚠️ Low metric consensus indicates uncertainty")
        
        return supporting, contradicting
    

    def _get_ai_indicator_from_metric(self, metric_name: str, result: MetricResult) -> Optional[str]:
        """
        Get AI indicator from metric result
        """
        details = result.details
        
        if (metric_name == "structural"):
            if (details.get("burstiness_score", 1.0) < 0.4):
                return self.AI_INDICATORS["low_burstiness"]

        elif (metric_name == "perplexity"):
            if (details.get("overall_perplexity", 100) < 35):
                return self.AI_INDICATORS["low_perplexity"]

        elif (metric_name == "entropy"):
            if (details.get("token_diversity", 1.0) < 0.65):
                return self.AI_INDICATORS["low_entropy"]

        elif (metric_name == "semantic_analysis"):
            if (details.get("coherence_score", 0.5) > 0.75):
                return self.AI_INDICATORS["semantic_perfection"]

        return None
    

    def _get_human_indicator_from_metric(self, metric_name: str, result: MetricResult) -> Optional[str]:
        """
        Get human indicator from metric result
        """
        details = result.details
        
        if (metric_name == "structural"):
            if (details.get("burstiness_score", 0.0) > 0.6):
                return self.HUMAN_INDICATORS["high_burstiness"]

        elif (metric_name == "perplexity"):
            if (details.get("overall_perplexity", 0) > 55):
                return self.HUMAN_INDICATORS["high_perplexity"]

        elif (metric_name == "entropy"):
            if (details.get("token_diversity", 0.0) > 0.75):
                return self.HUMAN_INDICATORS["high_entropy"]

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
                          }
        
        threshold_note  = threshold_info.get(domain, "Standard detection thresholds applied")
        
        return f"**Domain Analysis ({domain.value})**\n\n{context}\n\n{threshold_note}"
    

    def _explain_ensemble_methodology(self, ensemble_result: EnsembleResult, ensemble_method: str) -> str:
        """
        Explain the ensemble methodology used
        """
        method_desc = self.ENSEMBLE_METHODS.get(ensemble_method, "advanced aggregation of multiple detection methods")
        
        explanation = f"**Ensemble Methodology**: {method_desc}\n\n"
        
        # Explain key top-5 metrics
        top_metrics = sorted(ensemble_result.metric_weights.items(), key = lambda x: x[1], reverse = True)[:5]
        
        if top_metrics:
            explanation += "**Top contributing metrics**:\n"
            for metric, weight in top_metrics:
                explanation += f"• {metric}: {weight:.1%} weight\n"
        
        # Add reasoning snippets if available
        if hasattr(ensemble_result, 'reasoning') and ensemble_result.reasoning:
            key_reasons = [r for r in ensemble_result.reasoning if not r.startswith('##')][:2]
            if key_reasons:
                explanation += "\n**Key ensemble factors**:\n"
                for reason in key_reasons:
                    explanation += f"• {reason}\n"
        
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
    

    def _generate_attribution_reasoning(self, attribution_result: AttributionResult) -> str:
        """
        Generate reasoning for model attribution
        """
        model       = attribution_result.predicted_model
        confidence  = attribution_result.confidence
        
        if ((model == AIModel.UNKNOWN) or (confidence < 0.3)):
            return "**Model Attribution**: Uncertain. Text patterns don't strongly match known AI model fingerprints."
        
        model_name  = model.value.replace("-", " ").replace("_", " ").title()
        
        reasoning   = f"**Attributed Model**: {model_name} (confidence: {confidence:.1%})\n\n"
        
        # Model characteristics
        model_chars = {AIModel.GPT_3_5: "Characteristic patterns: frequent transitions, consistent structure, balanced explanations.",
                       AIModel.GPT_4: "Advanced patterns: sophisticated vocabulary, nuanced analysis, well-structured arguments.", 
                       AIModel.CLAUDE_3_OPUS: "Distinctive style: thoughtful analysis, balanced perspectives, explanatory depth.",
                       AIModel.GEMINI_PRO: "Typical patterns: conversational tone, clear explanations, exploratory language.",
                       AIModel.LLAMA_3: "Common traits: direct explanations, structured responses, consistent formatting.",
                      }
        
        reasoning  += model_chars.get(model, "Shows characteristic AI writing patterns.")
        
        # Add fingerprint matches if available
        if attribution_result.fingerprint_matches:
            reasoning += "\n\n**Top fingerprint matches**:"
            
            for model_name, score in list(attribution_result.fingerprint_matches.items())[:3]:
                reasoning += f"\n• {model_name}: {score}% match"
        
        return reasoning
    

    def _generate_ensemble_recommendations(self, ensemble_result: EnsembleResult, metric_results: Dict[str, MetricResult], domain: Domain) -> List[str]:
        """
        Generate actionable recommendations based on ensemble results
        """
        recommendations = list()
        verdict         = ensemble_result.final_verdict
        confidence      = ensemble_result.overall_confidence
        uncertainty     = ensemble_result.uncertainty_score
        
        # Base recommendations by verdict and confidence
        if ("AI-Generated" in verdict):
            if (confidence >= 0.8):
                rec = "**High-confidence AI detection**: Consider verified original drafts or alternative assessment methods."

            else:
                rec = "**Likely AI involvement**: Recommend discussion about AI tool usage and verification of understanding."
            
            recommendations.append(rec)
            
        elif ("Human-Written" in verdict):
            if (confidence >= 0.8):
                rec = "**High-confidence human authorship**: No additional verification typically needed."
            
            else:
                rec = "**Likely human-written**: Consider context and writing history for complete assessment."
            
            recommendations.append(rec)
            
        elif ("Mixed" in verdict):
            recommendations.append("**Mixed AI-human content**: Common in collaborative writing. Discuss appropriate AI use guidelines.")
        
        # Uncertainty-based recommendations
        if (uncertainty > 0.6):
            recommendations.append("**High uncertainty case**: Consider complementary verification methods like oral discussion or process documentation.")
        
        # Domain-specific recommendations
        domain_recs = {Domain.ACADEMIC      : "For academic work: verify subject mastery through targeted questions or practical application.",
                       Domain.CREATIVE      : "For creative work: assess originality, personal voice, and creative process documentation.", 
                       Domain.TECHNICAL_DOC : "For technical content: verify practical expertise and problem-solving ability.",
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
__all__ = ["DetailedReasoning", 
           "ReasoningGenerator",
          ]