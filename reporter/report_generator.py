# DEPENDENCIES
import json
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from detector.orchestrator import DetectionResult
from detector.attribution import AttributionResult
from reporter.reasoning_generator import DetailedReasoning
from reporter.reasoning_generator import ReasoningGenerator


@dataclass
class DetailedMetric:
    """
    Metric data structure with sub-metrics
    """
    name              : str
    ai_probability    : float
    human_probability : float
    confidence        : float
    verdict           : str
    description       : str
    detailed_metrics  : Dict[str, float]
    weight            : float


class ReportGenerator:
    """
    Generates comprehensive detection reports with detailed metrics
    
    Supports:
    - JSON (structured data with all details)
    - PDF (printable reports with tables and formatting)
    """
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator
        
        Arguments:
        ----------
            output_dir { str } : Directory for saving reports (default: data/reports)
        """
        if (output_dir is None):
            output_dir = Path(__file__).parent.parent / "data" / "reports"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents  = True, 
                              exist_ok = True,
                             )
        
        self.reasoning_generator = ReasoningGenerator()
        
        logger.info(f"ReportGenerator initialized (output_dir={self.output_dir})")
    

    def generate_complete_report(self, detection_result: DetectionResult, attribution_result: Optional[AttributionResult] = None, highlighted_sentences: Optional[List] = None, 
                                 formats: List[str] = ["json", "pdf"], filename_prefix: str = "ai_detection_report") -> Dict[str, str]:
        """
        Generate comprehensive report in JSON and PDF formats with detailed metrics
        
        Arguments:
        ----------
            detection_result      : Detection analysis result

            attribution_result    : Model attribution result (optional)

            highlighted_sentences : List of highlighted sentences (optional)
        
            formats               : List of formats to generate (json, pdf)
            
            filename_prefix       : Prefix for output filenames
            
        Returns:
        --------
                { dict }          : Dictionary mapping format to filepath
        """
        # Generate detailed reasoning
        reasoning        = self.reasoning_generator.generate(ensemble_result    = detection_result.ensemble_result,
                                                             metric_results     = detection_result.metric_results,
                                                             domain             = detection_result.domain_prediction.primary_domain,
                                                             attribution_result = attribution_result,
                                                             text_length        = detection_result.processed_text.word_count,
                                                            )
        
        # Extract detailed metrics from ACTUAL detection results
        detailed_metrics = self._extract_detailed_metrics(detection_result)
        
        # Timestamp for filenames
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files  = dict()
        
        # Generate requested formats
        if ("json" in formats):  
            json_path               = self._generate_json_report(detection_result      = detection_result, 
                                                                 reasoning             = reasoning, 
                                                                 detailed_metrics      = detailed_metrics, 
                                                                 attribution_result    = attribution_result,
                                                                 highlighted_sentences = highlighted_sentences,
                                                                 filename              = f"{filename_prefix}_{timestamp}.json",
                                                                )
            generated_files["json"] = str(json_path)
        
        if ("pdf" in formats):
            try:
                pdf_path               = self._generate_pdf_report(detection_result      = detection_result, 
                                                                   reasoning             = reasoning, 
                                                                   detailed_metrics      = detailed_metrics, 
                                                                   attribution_result    = attribution_result,
                                                                   highlighted_sentences = highlighted_sentences,
                                                                   filename              = f"{filename_prefix}_{timestamp}.pdf",
                                                                  )  
                generated_files["pdf"] = str(pdf_path)

            except Exception as e:
                logger.warning(f"PDF generation failed: {repr(e)}")
                logger.info("Install reportlab for PDF support: pip install reportlab")
        
        logger.info(f"Generated {len(generated_files)} report(s): {list(generated_files.keys())}")
        
        return generated_files


    def _extract_detailed_metrics(self, detection_result: DetectionResult) -> List[DetailedMetric]:
        """
        Extract detailed metrics with sub-metrics from ACTUAL detection result
        """
        detailed_metrics = list()
        metric_results   = detection_result.metric_results
        ensemble_result  = detection_result.ensemble_result
        
        # Get actual metric weights from ensemble
        metric_weights   = getattr(ensemble_result, 'metric_weights', {})
        
        # Extract actual metric data
        for metric_name, metric_result in metric_results.items():
            if metric_result.error is not None:
                continue
                
            # Get actual probabilities and confidence
            ai_prob    = metric_result.ai_probability * 100
            human_prob = metric_result.human_probability * 100
            confidence = metric_result.confidence * 100
            
            # Determine verdict based on actual probability
            if (ai_prob >= 60):
                verdict = "AI"

            elif (ai_prob <= 40):
                verdict = "HUMAN"

            else:
                verdict = "MIXED (AI + HUMAN)"
            
            # Get actual weight or use default
            weight                = metric_weights.get(metric_name, 0.0) * 100
            
            # Extract actual detailed metrics from metric result
            detailed_metrics_data = self._extract_metric_details(metric_name   = metric_name, 
                                                                 metric_result = metric_result,
                                                                )
            
            # Get description based on metric type
            description           = self._get_metric_description(metric_name = metric_name)
            
            detailed_metrics.append(DetailedMetric(name              = metric_name,
                                                   ai_probability    = ai_prob,
                                                   human_probability = human_prob,
                                                   confidence        = confidence,
                                                   verdict           = verdict,
                                                   description       = description,
                                                   detailed_metrics  = detailed_metrics_data,
                                                   weight            = weight,
                                                  )
                                   )
        
        return detailed_metrics


    def _extract_metric_details(self, metric_name: str, metric_result) -> Dict[str, float]:
        """
        Extract detailed sub-metrics from metric result
        """
        details = dict()
        
        # Try to get details from metric result
        if ((hasattr(metric_result, 'details')) and metric_result.details):
            details = metric_result.details.copy()
        
        # If no details available, provide basic calculated values
        if not details:
            details = {"ai_probability"    : metric_result.ai_probability * 100,
                       "human_probability" : metric_result.human_probability * 100,
                       "confidence"        : metric_result.confidence * 100,
                       "score"             : getattr(metric_result, 'score', 0.0) * 100,
                      }
        
        return details


    def _get_metric_description(self, metric_name: str) -> str:
        """
        Get description for each metric type
        """
        descriptions = {"structural"        : "Analyzes sentence structure, length patterns, and statistical features",
                        "perplexity"        : "Measures text predictability using language model cross-entropy",
                        "entropy"           : "Evaluates token diversity and sequence unpredictability",
                        "semantic_analysis" : "Examines semantic coherence, topic consistency, and logical flow",
                        "linguistic"        : "Assesses grammatical patterns, syntactic complexity, and style markers",
                        "detect_gpt"        : "Tests text stability under perturbation using curvature analysis",
                       }

        return descriptions.get(metric_name, "Advanced text analysis metric.")


    def _generate_json_report(self, detection_result: DetectionResult, reasoning: DetailedReasoning, detailed_metrics: List[DetailedMetric], 
                              attribution_result: Optional[AttributionResult], highlighted_sentences: Optional[List] = None, filename: str = None) -> Path:
        """
        Generate JSON format report with detailed metrics
        """
        # Convert metrics to serializable format
        metrics_data = list()

        for metric in detailed_metrics:
            metrics_data.append({"name"              : metric.name,
                                 "ai_probability"    : metric.ai_probability,
                                 "human_probability" : metric.human_probability,
                                 "confidence"        : metric.confidence,
                                 "verdict"           : metric.verdict,
                                 "description"       : metric.description,
                                 "weight"            : metric.weight,
                                 "detailed_metrics"  : metric.detailed_metrics,
                               })
        
        # Convert highlighted sentences to serializable format
        highlighted_data = None
        
        if highlighted_sentences:
            highlighted_data = list()

            for sent in highlighted_sentences:
                highlighted_data.append({"text"           : sent.text,
                                         "ai_probability" : sent.ai_probability,
                                         "confidence"     : sent.confidence,
                                         "color_class"    : sent.color_class,
                                         "index"          : sent.index,
                                       })

        # Attribution data - use attribution_result
        attribution_data = None
        
        if attribution_result:
            attribution_data = {"predicted_model"     : attribution_result.predicted_model.value,
                                "confidence"          : attribution_result.confidence,
                                "model_probabilities" : attribution_result.model_probabilities,
                                "reasoning"           : attribution_result.reasoning,
                                "fingerprint_matches" : attribution_result.fingerprint_matches,
                                "domain_used"         : attribution_result.domain_used.value,
                                "metric_contributions": attribution_result.metric_contributions,
                               }
        
        # Use ACTUAL detection results with ensemble integration
        ensemble_result = detection_result.ensemble_result
        
        report_data     = {"report_metadata"     : {"generated_at" : datetime.now().isoformat(),
                                                    "version"      : "1.0.0",
                                                    "format"       : "json",
                                                    "report_id"    : filename.replace('.json', ''),
                                                   },
                           "overall_results"     : {"final_verdict"      : ensemble_result.final_verdict,
                                                    "ai_probability"     : round(ensemble_result.ai_probability, 4),
                                                    "human_probability"  : round(ensemble_result.human_probability, 4),
                                                    "mixed_probability"  : round(ensemble_result.mixed_probability, 4),
                                                    "overall_confidence" : round(ensemble_result.overall_confidence, 4),
                                                    "uncertainty_score"  : round(ensemble_result.uncertainty_score, 4),
                                                    "consensus_level"    : round(ensemble_result.consensus_level, 4),
                                                    "domain"             : detection_result.domain_prediction.primary_domain.value,
                                                    "domain_confidence"  : round(detection_result.domain_prediction.confidence, 4),
                                                    "text_length"        : detection_result.processed_text.word_count,
                                                    "sentence_count"     : detection_result.processed_text.sentence_count,
                                                   },
                           "ensemble_analysis"   : {"method_used"     : "confidence_calibrated",
                                                    "metric_weights"  : {name: round(weight, 4) for name, weight in ensemble_result.metric_weights.items()},
                                                    "weighted_scores" : {name: round(score, 4) for name, score in ensemble_result.weighted_scores.items()},
                                                    "reasoning"       : ensemble_result.reasoning,
                                                   },
                           "detailed_metrics"    : metrics_data,
                           "detection_reasoning" : {"summary"                : reasoning.summary,
                                                    "key_indicators"         : reasoning.key_indicators,
                                                    "metric_explanations"    : reasoning.metric_explanations,
                                                    "supporting_evidence"    : reasoning.supporting_evidence,
                                                    "contradicting_evidence" : reasoning.contradicting_evidence,
                                                    "confidence_explanation" : reasoning.confidence_explanation,
                                                    "domain_analysis"        : reasoning.domain_analysis,
                                                    "ensemble_analysis"      : reasoning.ensemble_analysis,
                                                    "uncertainty_analysis"   : reasoning.uncertainty_analysis,
                                                    "recommendations"        : reasoning.recommendations,
                                                   },
                           "highlighted_text"    : highlighted_data,
                           "model_attribution"   : attribution_data,
                           "performance_metrics" : {"total_processing_time"  : round(detection_result.processing_time, 3),
                                                    "metrics_execution_time" : {name: round(time, 3) for name, time in detection_result.metrics_execution_time.items()},
                                                    "warnings"               : detection_result.warnings,
                                                    "errors"                 : detection_result.errors,
                                                   }
                          }
        
        output_path     = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(obj          = report_data, 
                      fp           = f, 
                      indent       = 4, 
                      ensure_ascii = False,
                     )
        
        logger.info(f"JSON report saved: {output_path}")
        return output_path


    def _generate_pdf_report(self, detection_result: DetectionResult, reasoning: DetailedReasoning, detailed_metrics: List[DetailedMetric], 
                             attribution_result: Optional[AttributionResult], highlighted_sentences: Optional[List] = None, filename: str = None) -> Path:
        """
        Generate PDF format report with detailed metrics
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        
        except ImportError:
            raise ImportError("reportlab is required for PDF generation. Install: pip install reportlab")
        
        output_path   = self.output_dir / filename
        
        # Create PDF
        doc           = SimpleDocTemplate(str(output_path),
                                          pagesize     = letter,
                                          rightMargin  = 50,
                                          leftMargin   = 50,
                                          topMargin    = 50,
                                          bottomMargin = 20,
                                         )
        
        # Container for PDF elements
        elements      = list()
        styles        = getSampleStyleSheet()
        
        # Custom styles
        title_style   = ParagraphStyle('CustomTitle',
                                       parent     = styles['Heading1'],
                                       fontSize   = 20,
                                       textColor  = colors.HexColor('#667eea'),
                                       spaceAfter = 20,
                                       alignment  = TA_CENTER,
                                      )
                                
        heading_style = ParagraphStyle('CustomHeading',
                                       parent      = styles['Heading2'],
                                       fontSize    = 14,
                                       textColor   = colors.HexColor('#111827'),
                                       spaceAfter  = 12,
                                       spaceBefore = 12,
                                      )
        
        body_style    = ParagraphStyle('CustomBody',
                                       parent     = styles['BodyText'],
                                       fontSize   = 10,
                                       alignment  = TA_JUSTIFY,
                                       spaceAfter = 8,
                                      )
        
        # Use detection results with ensemble integration
        ensemble_result = detection_result.ensemble_result
        
        # Title and main sections
        elements.append(Paragraph("AI Text Detection Analysis Report", title_style))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Verdict section with ensemble metrics
        elements.append(Paragraph("Detection Summary", heading_style))
        verdict_data = [['Final Verdict:', ensemble_result.final_verdict],
                        ['AI Probability:', f"{ensemble_result.ai_probability:.1%}"],
                        ['Human Probability:', f"{ensemble_result.human_probability:.1%}"],
                        ['Mixed Probability:', f"{ensemble_result.mixed_probability:.1%}"],
                        ['Overall Confidence:', f"{ensemble_result.overall_confidence:.1%}"],
                        ['Uncertainty Score:', f"{ensemble_result.uncertainty_score:.1%}"],
                        ['Consensus Level:', f"{ensemble_result.consensus_level:.1%}"],
                       ]
        
        verdict_table = Table(verdict_data, colWidths=[2*inch, 3*inch])
        verdict_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
                                           ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                           ('FONTSIZE', (0, 0), (-1, -1), 10),
                                           ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                         ])
                              )
        
        elements.append(verdict_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Content analysis
        elements.append(Paragraph("Content Analysis", heading_style))
        content_data = [['Content Domain:', detection_result.domain_prediction.primary_domain.value.title()],
                        ['Domain Confidence:', f"{detection_result.domain_prediction.confidence:.1%}"],
                        ['Word Count:', str(detection_result.processed_text.word_count)],
                        ['Sentence Count:', str(detection_result.processed_text.sentence_count)],
                        ['Processing Time:', f"{detection_result.processing_time:.2f}s"],
                       ]
        
        content_table = Table(content_data, colWidths=[2*inch, 3*inch])
        content_table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 10),
                                           ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                         ])
                              )
        
        elements.append(content_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Ensemble Analysis
        elements.append(Paragraph("Ensemble Analysis", heading_style))
        elements.append(Paragraph(f"Method: Confidence Calibrated Aggregation", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Metric weights table
        if hasattr(ensemble_result, 'metric_weights') and ensemble_result.metric_weights:
            elements.append(Paragraph("Metric Weights", styles['Heading3']))
            weight_data = [['Metric', 'Weight']]
            for metric, weight in ensemble_result.metric_weights.items():
                weight_data.append([metric.title(), f"{weight:.1%}"])
            
            weight_table = Table(weight_data, colWidths=[3*inch, 1*inch])
            weight_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                              ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                              ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                              ('FONTSIZE', (0, 0), (-1, -1), 9),
                                              ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                              ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                            ])
                                 )
            elements.append(weight_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Detailed metrics
        elements.append(Paragraph("Detailed Metric Analysis", heading_style))
        for metric in detailed_metrics:
            elements.append(Paragraph(f"{metric.name.title().replace('_', ' ')}", styles['Heading3']))
            metric_data = [['Verdict:', metric.verdict],
                           ['AI Probability:', f"{metric.ai_probability:.1f}%"],
                           ['Human Probability:', f"{metric.human_probability:.1f}%"],
                           ['Confidence:', f"{metric.confidence:.1f}%"],
                           ['Ensemble Weight:', f"{metric.weight:.1f}%"],
                          ]
            
            metric_table = Table(metric_data, colWidths=[1.5*inch, 1.5*inch])
            metric_table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 9),
                                              ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                            ])
                                 )
            
            elements.append(metric_table)
            elements.append(Paragraph(metric.description, body_style))
            
            # Add detailed sub-metrics if available
            if metric.detailed_metrics:
                elements.append(Paragraph("Detailed Metrics:", styles['Heading4']))
                sub_metric_data = [['Metric', 'Value']]
                for sub_name, sub_value in list(metric.detailed_metrics.items())[:6]:  # Show top 6
                    sub_metric_data.append([sub_name.replace('_', ' ').title(), f"{sub_value:.2f}"])
                
                sub_metric_table = Table(sub_metric_data, colWidths=[2*inch, 1*inch])
                sub_metric_table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 8),
                                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                                      ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                                                    ])
                                         )

                elements.append(sub_metric_table)
            
            elements.append(Spacer(1, 0.1*inch))
        
        # Detection Reasoning
        elements.append(Paragraph("Detection Reasoning", heading_style))
        elements.append(Paragraph(reasoning.summary, body_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Key Indicators
        elements.append(Paragraph("Key Indicators", styles['Heading3']))
        for indicator in reasoning.key_indicators[:5]:  # Show top 5
            elements.append(Paragraph(f"• {indicator}", body_style))
        
        elements.append(Spacer(1, 0.1*inch))
        
        # Confidence Explanation
        elements.append(Paragraph("Confidence Analysis", styles['Heading3']))
        elements.append(Paragraph(reasoning.confidence_explanation, body_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Uncertainty Analysis
        elements.append(Paragraph("Uncertainty Analysis", styles['Heading3']))
        elements.append(Paragraph(reasoning.uncertainty_analysis, body_style))
        
        # Model Attribution Section
        if attribution_result:
            elements.append(PageBreak())
            elements.append(Paragraph("AI Model Attribution", heading_style))
            
            # Attribution summary
            predicted_model = attribution_result.predicted_model.value.replace("_", " ").title()
            confidence = attribution_result.confidence * 100
            
            attribution_summary = [['Predicted Model:', predicted_model],
                                   ['Attribution Confidence:', f"{confidence:.1f}%"],
                                   ['Domain Used:', attribution_result.domain_used.value.title()],
                                  ]
            
            attribution_table = Table(attribution_summary, colWidths=[2*inch, 3*inch])
            attribution_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
                                                   ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                                   ('FONTSIZE', (0, 0), (-1, -1), 10),
                                                   ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                                 ])
                                      )
            
            elements.append(attribution_table)
            elements.append(Spacer(1, 0.1*inch))
            
            # Model probabilities table
            if attribution_result.model_probabilities:
                elements.append(Paragraph("Model Probability Breakdown", styles['Heading3']))
                
                prob_data     = [['Model', 'Probability']]
                
                # Show top 5
                sorted_models = sorted(attribution_result.model_probabilities.items(),
                                       key     = lambda x: x[1],
                                       reverse = True)[:5]
                
                for model_name, probability in sorted_models:
                    display_name = model_name.replace("_", " ").replace("-", " ").title()
                    prob_data.append([display_name, f"{probability:.1%}"])
                
                prob_table = Table(prob_data, colWidths=[3*inch, 1*inch])
                prob_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                                              ])
                                   )
                
                elements.append(prob_table)
                elements.append(Spacer(1, 0.2*inch))
            
            # Attribution reasoning
            if attribution_result.reasoning:
                elements.append(Paragraph("Attribution Reasoning", styles['Heading3']))
                for reason in attribution_result.reasoning[:3]:  # Show top 3 reasons
                    elements.append(Paragraph(f"• {reason}", body_style))
        
        # Recommendations
        elements.append(PageBreak())
        elements.append(Paragraph("Recommendations", heading_style))
        for recommendation in reasoning.recommendations:
            elements.append(Paragraph(f"• {recommendation}", body_style))
        
        # Footer
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(f"Generated by AI Text Detector v2.0 | Processing Time: {detection_result.processing_time:.2f}s", 
                                ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.gray)))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"PDF report saved: {output_path}")
        return output_path



# Export
__all__ = ["ReportGenerator", 
           "DetailedMetric",
          ]