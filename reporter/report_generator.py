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
        # Convert DetectionResult to dict for consistent access
        detection_dict = detection_result.to_dict() if hasattr(detection_result, 'to_dict') else detection_result
        
        # DEBUG: Check structure
        logger.debug(f"detection_dict keys: {list(detection_dict.keys())}")
        logger.debug(f"detection_dict type: {type(detection_dict)}")
        
        # Check if this is the full analysis result or just partial data
        # From your logs, it seems like during report generation, you're getting
        # a different/shorter text than the original analysis
        
        # Extract the actual detection data from the structure
        if ("detection_result" in detection_dict):
            detection_data = detection_dict["detection_result"]
            logger.debug("Extracted detection_result from outer dict")
            logger.debug(f"Detection data has analysis keys: {list(detection_data.get('analysis', {}).keys())}")
            logger.debug(f"Text length in analysis: {detection_data.get('analysis', {}).get('text_length', 'Unknown')}")
        else:
            detection_data = detection_dict
            logger.debug("Using detection_dict directly")
            logger.debug(f"Detection data has analysis keys: {list(detection_data.get('analysis', {}).keys())}")
        
        # Validate we have the correct data (not the short text)
        analysis_data = detection_data.get("analysis", {})
        text_length = analysis_data.get("text_length", 0)
        
        if text_length < 50:  # Less than minimum
            logger.warning(f"WARNING: Report generation received short text ({text_length} chars). "
                         f"This might be partial data instead of full analysis results.")
        
        # Generate detailed reasoning
        reasoning        = self.reasoning_generator.generate(ensemble_result    = detection_result.ensemble_result,
                                                             metric_results     = detection_result.metric_results,
                                                             domain             = detection_result.domain_prediction.primary_domain,
                                                             attribution_result = attribution_result,
                                                             text_length        = detection_result.processed_text.word_count,
                                                            )
        
        # Extract detailed metrics from ACTUAL detection results
        detailed_metrics = self._extract_detailed_metrics(detection_data)
        
        # Timestamp for filenames
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files  = dict()
        
        # Generate requested formats
        if ("json" in formats):  
            json_path               = self._generate_json_report(detection_data        = detection_data, 
                                                                 detection_dict_full   = detection_dict,
                                                                 reasoning             = reasoning, 
                                                                 detailed_metrics      = detailed_metrics, 
                                                                 attribution_result    = attribution_result,
                                                                 highlighted_sentences = highlighted_sentences,
                                                                 filename              = f"{filename_prefix}_{timestamp}.json",
                                                                )
            generated_files["json"] = str(json_path)
        
        if ("pdf" in formats):
            try:
                pdf_path               = self._generate_pdf_report(detection_data        = detection_data,
                                                                   detection_dict_full   = detection_dict,
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


    def _extract_detailed_metrics(self, detection_data: Dict) -> List[DetailedMetric]:
        """
        Extract detailed metrics with sub-metrics from ACTUAL detection result
        """
        detailed_metrics = list()
        metrics_data     = detection_data.get("metrics", {})
        ensemble_data    = detection_data.get("ensemble", {})
        
        # Get actual metric weights from ensemble
        metric_weights   = ensemble_data.get("metric_contributions", {})
        
        # Log what we're working with
        logger.debug(f"Extracting metrics from {len(metrics_data)} metrics")
        logger.debug(f"Metric names: {list(metrics_data.keys())}")
        
        # Extract actual metric data
        for metric_name, metric_result in metrics_data.items():
            if (not isinstance(metric_result, dict)):
                logger.warning(f"Metric {metric_name} is not a dict: {type(metric_result)}")
                continue
                
            error_msg = metric_result.get("error")
            if (error_msg is not None):
                logger.warning(f"Metric {metric_name} has error: {error_msg}")
                continue
                
            # Get actual probabilities and confidence
            ai_prob    = metric_result.get("ai_probability", 0)
            human_prob = metric_result.get("human_probability", 0)
            confidence = metric_result.get("confidence", 0)
            
            # DEBUG: Log extracted values
            logger.debug(f"Metric {metric_name}: AI={ai_prob}, Human={human_prob}, Confidence={confidence}")
            
            # Determine verdict based on actual probability
            # 60% threshold in decimal
            if (ai_prob >= 0.6):  
                verdict = "AI"
            
            # 40% threshold in decimal
            elif (ai_prob <= 0.4):  
                verdict = "HUMAN"

            else:
                verdict = "MIXED"
            
            # Get actual weight or use default
            weight = 0.0

            if metric_name in metric_weights:
                weight = metric_weights[metric_name].get("weight", 0.0)
            
            # Extract actual detailed metrics from metric result
            detailed_metrics_data = self._extract_metric_details(metric_name   = metric_name, 
                                                                 metric_result = metric_result,
                                                                )
            
            # Get description based on metric type
            description           = self._get_metric_description(metric_name = metric_name)
            
            detailed_metrics.append(DetailedMetric(name              = metric_name,
                                                   ai_probability    = ai_prob * 100,         # Convert to percentage
                                                   human_probability = human_prob * 100,      # Convert to percentage
                                                   confidence        = confidence * 100,      # Convert to percentage
                                                   verdict           = verdict,
                                                   description       = description,
                                                   detailed_metrics  = detailed_metrics_data,
                                                   weight            = weight * 100,          # Convert to percentage
                                                  )
                                   )
        
        logger.debug(f"Extracted {len(detailed_metrics)} detailed metrics")
        return detailed_metrics


    def _extract_metric_details(self, metric_name: str, metric_result: Dict) -> Dict[str, float]:
        """
        Extract detailed sub-metrics from metric result
        """
        details = dict()
        
        # Try to get details from metric result
        if metric_result.get("details"):
            details = metric_result["details"].copy()
        
        # If no details available, provide basic calculated values
        if not details:
            details = {"ai_probability"    : metric_result.get("ai_probability", 0) * 100,
                       "human_probability" : metric_result.get("human_probability", 0) * 100,
                       "confidence"        : metric_result.get("confidence", 0) * 100,
                       "score"             : metric_result.get("score", 0) * 100,
                      }
        
        return details


    def _get_metric_description(self, metric_name: str) -> str:
        """
        Get description for each metric type
        """
        descriptions = {"structural"                   : "Analyzes sentence structure, length patterns, and statistical features",
                        "perplexity"                   : "Measures text predictability using language model cross-entropy",
                        "entropy"                      : "Evaluates token diversity and sequence unpredictability",
                        "semantic_analysis"            : "Examines semantic coherence, topic consistency, and logical flow",
                        "linguistic"                   : "Assesses grammatical patterns, syntactic complexity, and style markers",
                        "multi_perturbation_stability" : "Tests text stability under perturbation using curvature analysis",
                       }

        return descriptions.get(metric_name, "Advanced text analysis metric.")


    def _generate_json_report(self, detection_data: Dict, detection_dict_full: Dict, reasoning: DetailedReasoning, detailed_metrics: List[DetailedMetric], 
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

        # Attribution data
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
        
        # Use ACTUAL detection results from dictionary
        ensemble_data        = detection_data.get("ensemble", {})
        analysis_data        = detection_data.get("analysis", {})
        metrics_data_dict    = detection_data.get("metrics", {})
        performance_data     = detection_data.get("performance", {})
        
        report_data          = {"report_metadata"     : {"generated_at" : datetime.now().isoformat(),
                                                         "version"      : "1.0.0",
                                                         "format"       : "json",
                                                         "report_id"    : filename.replace('.json', ''),
                                                        },
                                "overall_results"     : {"final_verdict"      : ensemble_data.get("final_verdict", "Unknown"),
                                                         "ai_probability"     : ensemble_data.get("ai_probability", 0),
                                                         "human_probability"  : ensemble_data.get("human_probability", 0),
                                                         "mixed_probability"  : ensemble_data.get("mixed_probability", 0),
                                                         "overall_confidence" : ensemble_data.get("overall_confidence", 0),
                                                         "uncertainty_score"  : ensemble_data.get("uncertainty_score", 0),
                                                         "consensus_level"    : ensemble_data.get("consensus_level", 0),
                                                         "domain"             : analysis_data.get("domain", "general"),
                                                         "domain_confidence"  : analysis_data.get("domain_confidence", 0),
                                                         "text_length"        : analysis_data.get("text_length", 0),
                                                         "sentence_count"     : analysis_data.get("sentence_count", 0),
                                                        },
                                "ensemble_analysis"   : {"method_used"     : "confidence_calibrated",
                                                         "metric_weights"  : ensemble_data.get("metric_contributions", {}),
                                                         "reasoning"       : ensemble_data.get("reasoning", []),
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
                                "performance_metrics" : {"total_processing_time"  : performance_data.get("total_time", 0),
                                                         "metrics_execution_time" : performance_data.get("metrics_time", {}),
                                                         "warnings"               : detection_data.get("warnings", []),
                                                         "errors"                 : detection_data.get("errors", []),
                                                        }
                               }
        
        output_path          = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(obj          = report_data, 
                      fp           = f, 
                      indent       = 4, 
                      ensure_ascii = False,
                     )
        
        logger.info(f"JSON report saved: {output_path}")
        return output_path


    def _generate_pdf_report(self, detection_data: Dict, detection_dict_full: Dict, reasoning: DetailedReasoning, detailed_metrics: List[DetailedMetric], 
                             attribution_result: Optional[AttributionResult], highlighted_sentences: Optional[List] = None, filename: str = None) -> Path:
        """
        Generate PDF format report with detailed metrics
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import Table
            from reportlab.lib.units import inch
            from reportlab.platypus import Spacer
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.enums import TA_LEFT
            from reportlab.platypus import PageBreak
            from reportlab.platypus import Paragraph
            from reportlab.lib.enums import TA_RIGHT
            from reportlab.graphics import renderPDF
            from reportlab.lib.enums import TA_CENTER
            from reportlab.platypus import TableStyle
            from reportlab.pdfgen.canvas import Canvas
            from reportlab.lib.enums import TA_JUSTIFY
            from reportlab.lib.pagesizes import letter
            from reportlab.graphics.shapes import Line
            from reportlab.graphics.shapes import Rect
            from reportlab.platypus import KeepTogether
            from reportlab.graphics.shapes import Circle
            from reportlab.graphics.shapes import Drawing
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate
            from reportlab.graphics.charts.piecharts import Pie
            from reportlab.platypus.flowables import HRFlowable
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.graphics.charts.textlabels import Label
            from reportlab.graphics.widgets.markers import makeMarker
          
        except ImportError:
            raise ImportError("reportlab is required for PDF generation. Install: pip install reportlab")
        
        output_path   = self.output_dir / filename
        
        # Create PDF with premium settings
        doc           = SimpleDocTemplate(str(output_path),
                                          pagesize     = A4,
                                          rightMargin  = 0.75*inch,
                                          leftMargin   = 0.75*inch,
                                          topMargin    = 0.75*inch,
                                          bottomMargin = 0.75*inch,
                                         )
        
        # Container for PDF elements
        elements          = list()
        styles            = getSampleStyleSheet()
        
        # Premium Color Scheme
        PRIMARY_COLOR     = colors.HexColor('#3b82f6')  # Blue-600
        SUCCESS_COLOR     = colors.HexColor('#10b981')  # Emerald-500
        WARNING_COLOR     = colors.HexColor('#f59e0b')  # Amber-500
        DANGER_COLOR      = colors.HexColor('#ef4444')  # Red-500
        INFO_COLOR        = colors.HexColor('#8b5cf6')  # Violet-500
        GRAY_LIGHT        = colors.HexColor('#f8fafc')  # Gray-50
        GRAY_MEDIUM       = colors.HexColor('#e2e8f0')  # Gray-200
        GRAY_DARK         = colors.HexColor('#334155')  # Gray-700
        TEXT_COLOR        = colors.HexColor('#1e293b')  # Gray-800
        
        # Helper function to get hex color string
        def get_hex_color(color_obj):
            """Convert color object to hex string"""
            if hasattr(color_obj, 'hexval'):
                return color_obj.hexval()
            elif hasattr(color_obj, 'rgb'):
                r, g, b = color_obj.rgb()
                return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            else:
                return "#3b82f6"  # Default blue
        
        # Premium Custom Styles
        title_style       = ParagraphStyle('PremiumTitle',
                                           parent     = styles['Heading1'],
                                           fontName   = 'Helvetica-Bold',
                                           fontSize   = 28,
                                           textColor  = PRIMARY_COLOR,
                                           spaceAfter = 20,
                                           alignment  = TA_CENTER,
                                          ) 
        
        subtitle_style    = ParagraphStyle('PremiumSubtitle',
                                           parent     = styles['Normal'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 12,
                                           textColor  = GRAY_DARK,
                                           spaceAfter = 30,
                                           alignment  = TA_CENTER,
                                          )
                                
        section_style     = ParagraphStyle('PremiumSection',
                                           parent      = styles['Heading2'],
                                           fontName    = 'Helvetica-Bold',
                                           fontSize    = 18,
                                           textColor   = TEXT_COLOR,
                                           spaceAfter  = 12,
                                           spaceBefore = 20,
                                          )
        
        subsection_style  = ParagraphStyle('PremiumSubSection',
                                           parent      = styles['Heading3'],
                                           fontName    = 'Helvetica-Bold',
                                           fontSize    = 14,
                                           textColor   = GRAY_DARK,
                                           spaceAfter  = 8,
                                           spaceBefore = 16,
                                          )
        
        body_style        = ParagraphStyle('PremiumBody',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 11,
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_JUSTIFY,
                                           spaceAfter = 8,
                                          )
        
        # Use detection results from detection_data
        ensemble_data     = detection_data.get("ensemble", {})
        analysis_data     = detection_data.get("analysis", {})
        performance_data  = detection_data.get("performance", {})
        
        # Extract values
        ai_prob           = ensemble_data.get("ai_probability", 0)
        human_prob        = ensemble_data.get("human_probability", 0)
        mixed_prob        = ensemble_data.get("mixed_probability", 0)
        confidence        = ensemble_data.get("overall_confidence", 0)
        uncertainty       = ensemble_data.get("uncertainty_score", 0)
        consensus         = ensemble_data.get("consensus_level", 0)
        final_verdict     = ensemble_data.get("final_verdict", "Unknown")
        
        # Determine colors based on verdict
        if ("Human".lower() in final_verdict.lower()):
            verdict_color = SUCCESS_COLOR
        elif ("AI".lower() in final_verdict.lower()):
            verdict_color = DANGER_COLOR
        elif ("Mixed".lower() in final_verdict.lower()):
            verdict_color = WARNING_COLOR
        else:
            verdict_color = PRIMARY_COLOR
        
        # Get hex strings for colors
        primary_hex = get_hex_color(PRIMARY_COLOR)
        success_hex = get_hex_color(SUCCESS_COLOR)
        warning_hex = get_hex_color(WARNING_COLOR)
        danger_hex = get_hex_color(DANGER_COLOR)
        info_hex = get_hex_color(INFO_COLOR)
        verdict_hex = get_hex_color(verdict_color)
        
        # Header
        elements.append(Paragraph("AI DETECTION ANALYTICS", 
                         ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=GRAY_DARK, alignment=TA_RIGHT)))
        
        elements.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=20))
        
        # Title and main sections
        elements.append(Paragraph("AI Text Detection Analysis Report", title_style))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
        
        # Add decorative line
        elements.append(HRFlowable(width="80%", thickness=2, color=PRIMARY_COLOR, spaceBefore=10, spaceAfter=30, hAlign='CENTER'))
        
        # Quick Stats Banner
        stats_data  = [['', 'AI', 'HUMAN', 'MIXED'],
                       ['Probability', f"{ai_prob:.1%}", f"{human_prob:.1%}", f"{mixed_prob:.1%}"]]
        
        stats_table = Table(stats_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (1, 1), (1, 1), DANGER_COLOR),
            ('BACKGROUND', (2, 1), (2, 1), SUCCESS_COLOR),
            ('BACKGROUND', (3, 1), (3, 1), WARNING_COLOR),
            ('TEXTCOLOR', (1, 1), (-1, 1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
            ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Main Verdict Section with colored badge
        elements.append(Paragraph("DETECTION VERDICT", section_style))
        
        verdict_box_data = [[
            Paragraph(f"<font size=18 color='{verdict_hex}'><b>{final_verdict.upper()}</b></font>", 
                     ParagraphStyle('VerdictText', alignment=TA_CENTER)),
            Paragraph(f"<font size=12>Confidence: <b>{confidence:.1%}</b></font><br/>"
                     f"<font size=10>Uncertainty: {uncertainty:.1%} | Consensus: {consensus:.1%}</font>",
                     ParagraphStyle('VerdictDetails', alignment=TA_CENTER))
        ]]
        
        verdict_box = Table(verdict_box_data, colWidths=[2.5*inch, 3*inch])
        verdict_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), GRAY_LIGHT),
            ('BACKGROUND', (1, 0), (1, 0), GRAY_LIGHT),
            ('BOX', (0, 0), (-1, -1), 1, verdict_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(verdict_box)
        elements.append(Spacer(1, 0.3*inch))
        
        # Content Analysis in a sleek table
        elements.append(Paragraph("CONTENT ANALYSIS", section_style))
        
        domain = analysis_data.get("domain", "general").title().replace('_', ' ')
        domain_confidence = analysis_data.get("domain_confidence", 0)
        text_length = analysis_data.get("text_length", 0)
        sentence_count = analysis_data.get("sentence_count", 0)
        total_time = performance_data.get("total_time", 0)
        
        # Create two-column layout for content analysis
        content_data = [
            [Paragraph("<b>Content Domain</b>", body_style), 
             Paragraph(f"<font color='{info_hex}'><b>{domain}</b></font> ({domain_confidence:.1%} confidence)", body_style)],
            [Paragraph("<b>Text Statistics</b>", body_style), 
             Paragraph(f"{text_length:,} words | {sentence_count:,} sentences", body_style)],
            [Paragraph("<b>Processing Time</b>", body_style), 
             Paragraph(f"{total_time:.2f} seconds", body_style)],
            [Paragraph("<b>Analysis Method</b>", body_style), 
             Paragraph("Confidence-Weighted Ensemble Aggregation", body_style)],
        ]
        
        content_table = Table(content_data, colWidths=[2*inch, 4*inch])
        content_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.25, GRAY_MEDIUM),
            ('BACKGROUND', (0, 0), (0, -1), GRAY_LIGHT),
        ]))
        
        elements.append(content_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Metric Weights Visualization
        elements.append(Paragraph("METRIC CONTRIBUTIONS", section_style))
        
        metric_contributions = ensemble_data.get("metric_contributions", {})
        if metric_contributions and len(metric_contributions) > 0:
            # Create horizontal bar chart effect with table
            weight_data = [['METRIC', 'WEIGHT', '']]
            
            for metric_name, contribution in metric_contributions.items():
                weight = contribution.get("weight", 0)
                display_name = metric_name.title().replace('_', ' ')
                
                # Create visual bar representation
                bar_width = int(weight * 50)  # Use 50 chars max for better fit
                bar_cell = f"[{'█' * bar_width}{'░' * (50-bar_width)}] {weight:.1%}"
                
                weight_data.append([display_name, f"{weight:.1%}", bar_cell])
            
            weight_table = Table(weight_data, colWidths=[2*inch, 1*inch, 2.5*inch])
            weight_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MEDIUM),
                ('TEXTCOLOR', (2, 1), (2, -1), PRIMARY_COLOR),
                ('FONTNAME', (2, 1), (2, -1), 'Courier'),
            ]))
            
            elements.append(weight_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Detailed Metric Analysis with colored cards
        elements.append(Paragraph("DETAILED METRIC ANALYSIS", section_style))
        
        if detailed_metrics:
            for metric in detailed_metrics:
                # Determine metric color based on verdict
                if metric.verdict == "HUMAN":
                    metric_color = SUCCESS_COLOR
                    prob_color = SUCCESS_COLOR
                elif metric.verdict == "AI":
                    metric_color = DANGER_COLOR
                    prob_color = DANGER_COLOR
                else:
                    metric_color = WARNING_COLOR
                    prob_color = WARNING_COLOR
                
                metric_color_hex = get_hex_color(metric_color)
                prob_color_hex = get_hex_color(prob_color)
                
                # Create metric card
                metric_card_data = [[
                    Paragraph(f"<font color='{metric_color_hex}' size=12><b>{metric.name.upper().replace('_', ' ')}</b></font><br/>"
                             f"<font size=9>{metric.description}</font>", 
                             ParagraphStyle('MetricTitle', alignment=TA_LEFT)),
                    
                    Paragraph(f"<font size=11><b>VERDICT</b></font><br/>"
                             f"<font color='{metric_color_hex}' size=12><b>{metric.verdict}</b></font>",
                             ParagraphStyle('MetricVerdict', alignment=TA_CENTER)),
                    
                    Paragraph(f"<font size=11><b>AI PROBABILITY</b></font><br/>"
                             f"<font color='{prob_color_hex}' size=12><b>{metric.ai_probability:.1f}%</b></font>",
                             ParagraphStyle('MetricProbability', alignment=TA_CENTER)),
                    
                    Paragraph(f"<font size=11><b>WEIGHT</b></font><br/>"
                             f"<font size=12><b>{metric.weight:.1f}%</b></font>",
                             ParagraphStyle('MetricWeight', alignment=TA_CENTER)),
                ]]
                
                metric_table = Table(metric_card_data, colWidths=[2.5*inch, 1.2*inch, 1.2*inch, 1*inch])
                metric_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), GRAY_LIGHT),
                    ('BOX', (0, 0), (-1, 0), 1, metric_color),
                    ('LINEABOVE', (0, 0), (-1, 0), 2, metric_color),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('TOPPADDING', (0, 0), (-1, 0), 10),
                ]))
                
                elements.append(metric_table)
                
                # Add detailed sub-metrics if available
                if metric.detailed_metrics:
                    elements.append(Spacer(1, 0.1*inch))
                    
                    # Create a grid of sub-metrics
                    sub_items = list(metric.detailed_metrics.items())[:6]
                    sub_data = []
                    
                    for i in range(0, len(sub_items), 3):
                        row = []
                        for j in range(3):
                            if i + j < len(sub_items):
                                sub_name, sub_value = sub_items[i + j]
                                # Format the value
                                if isinstance(sub_value, (int, float)):
                                    if sub_name.endswith('_score') or sub_name.endswith('_probability'):
                                        formatted_value = f"{sub_value:.1f}%"
                                    elif sub_name.endswith('_ratio') or sub_name.endswith('_frequency'):
                                        formatted_value = f"{sub_value:.3f}"
                                    elif sub_name.endswith('_entropy') or sub_name.endswith('_perplexity'):
                                        formatted_value = f"{sub_value:.2f}"
                                    else:
                                        formatted_value = f"{sub_value:.2f}"
                                else:
                                    formatted_value = str(sub_value)
                                
                                row.append(f"<b>{sub_name.replace('_', ' ').title()}:</b> {formatted_value}")
                            else:
                                row.append("")
                        
                        sub_data.append(row)
                    
                    if sub_data:
                        sub_table = Table(sub_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])
                        sub_table.setStyle(TableStyle([
                            ('FONTSIZE', (0, 0), (-1, -1), 8),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                            ('TOPPADDING', (0, 0), (-1, -1), 4),
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ]))
                        elements.append(sub_table)
                
                elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("No detailed metrics available for this analysis.", body_style))
            elements.append(Spacer(1, 0.2*inch))
        
        # Detection Reasoning
        elements.append(Paragraph("DETECTION REASONING", section_style))
        
        # Summary in a colored box
        summary_box = Table([[Paragraph(f"<font size=11>{reasoning.summary}</font>", body_style)]], colWidths=[6.5*inch])
        summary_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), GRAY_LIGHT),
            ('BOX', (0, 0), (-1, -1), 1, PRIMARY_COLOR),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(summary_box)
        elements.append(Spacer(1, 0.2*inch))
        
        # Key Indicators
        if reasoning.key_indicators:
            elements.append(Paragraph("KEY INDICATORS", subsection_style))
            
            indicators_data = []
            for i in range(0, len(reasoning.key_indicators), 2):
                row = []
                for j in range(2):
                    if i + j < len(reasoning.key_indicators):
                        indicator = reasoning.key_indicators[i + j]
                        # Add checkmark for positive indicators
                        if indicator.startswith("✅") or indicator.startswith("✓"):
                            icon_color = success_hex
                        elif indicator.startswith("⚠️") or indicator.startswith("❌"):
                            icon_color = warning_hex
                        else:
                            icon_color = primary_hex
                        
                        row.append(Paragraph(f"<font color='{icon_color}'>•</font> {indicator}", body_style))
                    else:
                        row.append("")
                indicators_data.append(row)
            
            indicators_table = Table(indicators_data, colWidths=[3*inch, 3*inch])
            indicators_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            
            elements.append(indicators_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Page break for attribution section
        elements.append(PageBreak())
        
        # Model Attribution Section
        if attribution_result:
            elements.append(Paragraph("AI MODEL ATTRIBUTION", section_style))
            
            predicted_model = attribution_result.predicted_model.value.replace("_", " ").title()
            attribution_confidence = attribution_result.confidence * 100
            
            attribution_card_data = [
                [Paragraph("<b>PREDICTED MODEL</b>", subsection_style), 
                 Paragraph(f"<font size=14 color='{info_hex}'><b>{predicted_model}</b></font>", subsection_style)],
                [Paragraph("<b>ATTRIBUTION CONFIDENCE</b>", subsection_style), 
                 Paragraph(f"<font size=14><b>{attribution_confidence:.1f}%</b></font>", subsection_style)],
                [Paragraph("<b>DOMAIN USED</b>", subsection_style), 
                 Paragraph(f"<b>{attribution_result.domain_used.value.title()}</b>", subsection_style)],
            ]
            
            attribution_table = Table(attribution_card_data, colWidths=[2.5*inch, 3.5*inch])
            attribution_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), GRAY_LIGHT),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MEDIUM),
            ]))
            
            elements.append(attribution_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # Model probabilities table
            if attribution_result.model_probabilities:
                elements.append(Paragraph("MODEL PROBABILITY DISTRIBUTION", subsection_style))
                
                prob_data = [['MODEL', 'PROBABILITY', '']]
                
                # Show top 6 models
                sorted_models = sorted(attribution_result.model_probabilities.items(),
                                       key=lambda x: x[1], reverse=True)[:6]
                
                for model_name, probability in sorted_models:
                    display_name = model_name.replace("_", " ").replace("-", " ").title()
                    bar_width = int(probability * 40)  # 40 chars max
                    prob_data.append([
                        display_name,
                        f"{probability:.1%}",
                        f"[{'█' * bar_width}{'░' * (40-bar_width)}]"
                    ])
                
                prob_table = Table(prob_data, colWidths=[2.5*inch, 1*inch, 2*inch])
                prob_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), INFO_COLOR),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MEDIUM),
                    ('FONTNAME', (2, 1), (2, -1), 'Courier'),
                    ('TEXTCOLOR', (2, 1), (2, -1), INFO_COLOR),
                ]))
                
                elements.append(prob_table)
                elements.append(Spacer(1, 0.3*inch))
        
        # Recommendations in colored boxes
        if reasoning.recommendations:
            elements.append(Paragraph("RECOMMENDATIONS", section_style))
            
            for i, recommendation in enumerate(reasoning.recommendations):
                # Alternate colors for visual interest
                if i % 3 == 0:
                    rec_color = success_hex
                elif i % 3 == 1:
                    rec_color = info_hex
                else:
                    rec_color = warning_hex
                
                rec_box = Table([[Paragraph(f"<font color='{rec_color}'>✓</font> {recommendation}", body_style)]], colWidths=[6.5*inch])
                rec_box.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), GRAY_LIGHT),
                    ('PADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMMARGIN', (0, 0), (-1, -1), 5),
                ]))
                
                elements.append(rec_box)
                elements.append(Spacer(1, 0.1*inch))
        
        # Footer with watermark
        footer_style = ParagraphStyle('FooterStyle',
                                     parent     = styles['Normal'],
                                     fontName   = 'Helvetica',
                                     fontSize   = 9,
                                     textColor  = GRAY_DARK,
                                     alignment  = TA_CENTER,
                                    )
        
        elements.append(Spacer(1, 0.5*inch))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_MEDIUM, spaceAfter=10))
        
        footer_text = (f"Generated by AI Text Detector v2.0 | "
                      f"Processing Time: {total_time:.2f}s | "
                      f"Report ID: {filename.replace('.pdf', '')}")
        
        elements.append(Paragraph(footer_text, footer_style))
        elements.append(Paragraph("Confidential Analysis Report • © 2025 AI Detection Analytics", 
                        ParagraphStyle('Copyright', parent=footer_style, fontSize=8, textColor=GRAY_MEDIUM)))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"Premium PDF report saved: {output_path}")
        return output_path



# Export
__all__ = ["ReportGenerator", 
           "DetailedMetric",
          ]