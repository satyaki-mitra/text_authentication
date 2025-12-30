# DEPENDENCIES
import re 
import json
from typing import Any
from typing import Dict
from typing import List
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime
from config.schemas import DetectionResult
from config.schemas import DetailedMetricResult
from config.schemas import DetailedReasoningResult
from services.reasoning_generator import ReasoningGenerator


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
    

    def generate_complete_report(self, detection_result: DetectionResult, highlighted_sentences: Optional[List] = None, formats: List[str] = ["json", "pdf"], 
                                 filename_prefix: str = "text_authenticity_report") -> Dict[str, str]:
        """
        Generate comprehensive report in JSON and PDF formats with detailed metrics
        
        Arguments:
        ----------
            detection_result      : Detection analysis result

            highlighted_sentences : List of highlighted sentences (optional)
        
            formats               : List of formats to generate (json, pdf)
            
            filename_prefix       : Prefix for output filenames
            
        Returns:
        --------
                { dict }          : Dictionary mapping format to filepath
        """
        # Convert DetectionResult to dict for consistent access
        detection_dict = detection_result.to_dict() if hasattr(detection_result, 'to_dict') else detection_result
        
        # Extract the actual detection data from the structure
        if ("detection_result" in detection_dict):
            detection_data = detection_dict["detection_result"]
            logger.info("Extracted detection_result from outer dict")

        else:
            detection_data = detection_dict
            logger.info("Using detection_dict directly")
        
        # Generate detailed reasoning
        reasoning        = self.reasoning_generator.generate(ensemble_result = detection_result.ensemble_result,
                                                             metric_results  = detection_result.metric_results,
                                                             domain          = detection_result.domain_prediction.primary_domain,
                                                             text_length     = detection_result.processed_text.word_count,
                                                            )
        
        # Extract detailed metrics from ACTUAL detection results
        detailed_metrics = self._extract_detailed_metrics(detection_data = detection_data)
        
        # Timestamp for filenames
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files  = dict()
        
        # Generate requested formats
        if ("json" in formats):  
            json_path               = self._generate_json_report(detection_data        = detection_data, 
                                                                 detection_dict_full   = detection_dict,
                                                                 reasoning             = reasoning, 
                                                                 detailed_metrics      = detailed_metrics,
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
                                                                   highlighted_sentences = highlighted_sentences,
                                                                   filename              = f"{filename_prefix}_{timestamp}.pdf",
                                                                  )  
                generated_files["pdf"] = str(pdf_path)

            except Exception as e:
                logger.warning(f"PDF generation failed: {repr(e)}")
                logger.info("Install reportlab for PDF support: pip install reportlab")
        
        logger.info(f"Generated {len(generated_files)} report(s): {list(generated_files.keys())}")
        
        return generated_files


    def _extract_detailed_metrics(self, detection_data: Dict) -> List[DetailedMetricResult]:
        """
        Extract detailed metrics with sub-metrics from ACTUAL detection result
        """
        detailed_metrics = list()
        metrics_data     = detection_data.get("metrics", {})
        ensemble_data    = detection_data.get("ensemble", {})
        
        # Get actual metric weights from ensemble
        metric_weights   = ensemble_data.get("metric_contributions", {})
        
        # Extract actual metric data
        for metric_name, metric_result in metrics_data.items():
            if (not isinstance(metric_result, dict)):
                logger.warning(f"Metric {metric_name} is not a dict: {type(metric_result)}")
                continue
                
            if (metric_result.get("error") is not None):
                logger.warning(f"Metric {metric_name} has error: {metric_result.get('error')}")
                continue
                
            # Get actual probabilities and confidence
            synthetic_prob = metric_result.get("synthetic_probability", 0)
            authentic_prob = metric_result.get("authentic_probability", 0)
            confidence = metric_result.get("confidence", 0)
            
            # Determine verdict based on actual probability
            if (authentic_prob >= 0.6):  
                verdict = "Authentically-Written"

            elif (synthetic_prob >= 0.6):  
                verdict = "Synthetically-Generated"
            
            elif (synthetic_prob > 0.4 and synthetic_prob < 0.6):
                verdict = "Hybrid"
            
            elif (authentic_prob > 0.4 and authentic_prob < 0.6):
                verdict = "Hybrid"
            
            else:
                # If both low, check which is higher
                if (authentic_prob > synthetic_prob):
                    verdict = "Authentically-Written"
                
                elif (synthetic_prob > authentic_prob):
                    verdict = "Synthetically-Generated"
                
                else:
                    verdict = "Hybrid"
            
            # Get actual weight or use default
            weight = 0.0
            if (metric_name in metric_weights):
                weight = metric_weights[metric_name].get("weight", 0.0)
            
            # Extract actual detailed metrics from metric result
            detailed_metrics_data = self._extract_metric_details(metric_name   = metric_name, 
                                                                 metric_result = metric_result,
                                                                )
            
            # Get description based on metric type
            description           = self._get_metric_description(metric_name = metric_name)
            
            detailed_metrics.append(DetailedMetricResult(name                  = metric_name,
                                                         synthetic_probability = synthetic_prob * 100,  # Convert to percentage
                                                         authentic_probability = authentic_prob * 100,  # Convert to percentage
                                                         confidence            = confidence * 100,      # Convert to percentage
                                                         verdict               = verdict,
                                                         description           = description,
                                                         detailed_metrics      = detailed_metrics_data,
                                                         weight                = weight * 100,          # Convert to percentage
                                                        )
                                   )
        
        logger.info(f"Extracted {len(detailed_metrics)} detailed metrics")
        
        return detailed_metrics


    def _extract_metric_details(self, metric_name: str, metric_result: Dict) -> Dict[str, float]:
        """
        Extract detailed sub-metrics from metric result
        """
        details = dict()
        
        # Try to get details from metric result
        if metric_result.get("details"):
            # Extract all numeric details
            for key, value in metric_result["details"].items():
                if (isinstance(value, (int, float))):
                    # Format specific metrics appropriately
                    if ("perplexity" in key.lower()):
                        details[key] = float(f"{value:.2f}")

                    elif ("entropy" in key.lower()):
                        details[key] = float(f"{value:.2f}")

                    elif (("score" in key.lower()) or ("ratio" in key.lower())):
                        details[key] = float(f"{value:.4f}")

                    elif ("probability" in key.lower()):
                        details[key] = float(f"{value:.4f}")

                    else:
                        details[key] = float(f"{value:.3f}")
                
                else:
                    details[key] = value
        
        # If no details available, provide basic calculated values
        if not details:
            details = {"synthetic_probability"    : metric_result.get("synthetic_probability", 0) * 100,
                       "authentic_probability" : metric_result.get("authentic_probability", 0) * 100,
                       "confidence"        : metric_result.get("confidence", 0) * 100,
                       "score"             : metric_result.get("raw_score", 0) * 100,
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


    def _generate_json_report(self, detection_data: Dict, detection_dict_full: Dict, reasoning: DetailedReasoningResult, detailed_metrics: List[DetailedMetricResult], 
                              highlighted_sentences: Optional[List] = None, filename: str = None) -> Path:
        """
        Generate JSON format report with detailed metrics
        """
        # Convert metrics to serializable format
        metrics_data = list()

        for metric in detailed_metrics:
            metrics_data.append({"name"              : metric.name,
                                 "synthetic_probability"    : metric.synthetic_probability,
                                 "authentic_probability" : metric.authentic_probability,
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
                                         "synthetic_probability" : sent.synthetic_probability,
                                         "confidence"     : sent.confidence,
                                         "color_class"    : sent.color_class,
                                         "index"          : sent.index,
                                       })
        
        # Use detection results from dictionary
        ensemble_data        = detection_data.get("ensemble", {})
        analysis_data        = detection_data.get("analysis", {})
        metrics_data_dict    = detection_data.get("metrics", {})
        performance_data     = detection_data.get("performance", {})
        
        report_data          = {"report_metadata"     : {"generated_at" : datetime.now().isoformat(),
                                                         "version"      : "1.0.0",
                                                         "format"       : "json",
                                                         "report_id"    : filename.replace('.json', ''),
                                                        },
                                "overall_results"     : {"final_verdict"         : ensemble_data.get("final_verdict", "Unknown"),
                                                         "synthetic_probability" : ensemble_data.get("synthetic_probability", 0),
                                                         "authentic_probability" : ensemble_data.get("authentic_probability", 0),
                                                         "hybrid_probability"     : ensemble_data.get("hybrid_probability", 0),
                                                         "overall_confidence"    : ensemble_data.get("overall_confidence", 0),
                                                         "uncertainty_score"     : ensemble_data.get("uncertainty_score", 0),
                                                         "consensus_level"       : ensemble_data.get("consensus_level", 0),
                                                         "domain"                : analysis_data.get("domain", "general"),
                                                         "domain_confidence"     : analysis_data.get("domain_confidence", 0),
                                                         "text_length"           : analysis_data.get("text_length", 0),
                                                         "sentence_count"        : analysis_data.get("sentence_count", 0),
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
    

    def _generate_pdf_report(self, detection_data: Dict, detection_dict_full: Dict, reasoning: DetailedReasoningResult, detailed_metrics: List[DetailedMetricResult], 
                             highlighted_sentences: Optional[List] = None, filename: str = None) -> Path:
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
        
        output_path       = self.output_dir / filename
        
        # Create PDF with pre-defined settings
        doc               = SimpleDocTemplate(str(output_path),
                                              pagesize     = A4,
                                              rightMargin  = 0.75*inch,
                                              leftMargin   = 0.75*inch,
                                              topMargin    = 0.75*inch,
                                              bottomMargin = 0.75*inch,
                                             )
        
        # Container for PDF elements
        elements          = list()
        styles            = getSampleStyleSheet()
        
        # Color Scheme
        PRIMARY_COLOR     = '#3b82f6' # Blue-600
        SUCCESS_COLOR     = '#10b981' # Emerald-500
        WARNING_COLOR     = '#f59e0b' # Amber-500
        DANGER_COLOR      = '#ef4444' # Red-500
        INFO_COLOR        = '#8b5cf6' # Violet-500
        GRAY_LIGHT        = '#f8fafc' # Gray-50
        GRAY_MEDIUM       = '#e2e8f0' # Gray-200
        GRAY_DARK         = '#334155' # Gray-700
        TEXT_COLOR        = '#1e293b' # Gray-800
        
        # Custom Styles
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
        
        filename_style    = ParagraphStyle('FilenameStyle',
                                           parent     = styles['Normal'],
                                           fontName   = 'Helvetica-Bold',
                                           fontSize   = 10,
                                           textColor  = GRAY_DARK,
                                           spaceAfter = 10,
                                           alignment  = TA_CENTER,
                                          )
                                
        section_style     = ParagraphStyle('PremiumSection',
                                           parent      = styles['Heading2'],
                                           fontName    = 'Helvetica-Bold',
                                           fontSize    = 18,
                                           textColor   = TEXT_COLOR,
                                           spaceAfter  = 12,
                                           spaceBefore = 20,
                                           underlineWidth = 1,
                                           underlineColor = PRIMARY_COLOR,
                                          )
        
        subsection_style  = ParagraphStyle('PremiumSubSection',
                                           parent      = styles['Heading3'],
                                           fontName    = 'Helvetica-Bold',
                                           fontSize    = 14,
                                           textColor   = GRAY_DARK,
                                           spaceAfter  = 8,
                                           spaceBefore = 16,
                                          )
        
        key_indicators_style = ParagraphStyle('KeyIndicatorsStyle',
                                              parent         = styles['Heading2'],
                                              fontName       = 'Helvetica-Bold',
                                              fontSize       = 18,
                                              textColor      = TEXT_COLOR,
                                              spaceAfter     = 12,
                                              spaceBefore    = 20,
                                              underlineWidth = 1,
                                              underlineColor = PRIMARY_COLOR,
                                             )
        
        body_style        = ParagraphStyle('PremiumBody',
                                          parent     = styles['BodyText'],
                                          fontName   = 'Helvetica',
                                          fontSize   = 11,
                                          textColor  = TEXT_COLOR,
                                          alignment  = TA_JUSTIFY,
                                          spaceAfter = 8,
                                         )
        
        # Larger font for page 2 content
        page2_body_style  = ParagraphStyle('Page2Body',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 11,  
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_JUSTIFY,
                                           spaceAfter = 8,
                                          )
        
        bullet_style      = ParagraphStyle('BulletStyle',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 11,
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_LEFT,
                                           spaceAfter = 6,
                                           leftIndent = 20,
                                          )
        
        bold_style        = ParagraphStyle('BoldStyle',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica-Bold',
                                           fontSize   = 11,
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_LEFT,
                                           spaceAfter = 8,
                                          )
        
        small_bold_style  = ParagraphStyle('SmallBoldStyle',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica-Bold',
                                           fontSize   = 9,
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_LEFT,
                                           spaceAfter = 4,
                                          )
        
        small_style       = ParagraphStyle('SmallStyle',
                                           parent     = styles['BodyText'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 9,
                                           textColor  = TEXT_COLOR,
                                           alignment  = TA_LEFT,
                                           spaceAfter = 4,
                                          )
        
        footer_style      = ParagraphStyle('FooterStyle',
                                           parent     = styles['Normal'],
                                           fontName   = 'Helvetica',
                                           fontSize   = 9,
                                           textColor  = GRAY_DARK,
                                           alignment  = TA_CENTER,
                                          ) 

        # Use detection results from detection_data
        ensemble_data     = detection_data.get("ensemble", {})
        analysis_data     = detection_data.get("analysis", {})
        performance_data  = detection_data.get("performance", {})
        
        # Extract filename from  file_info 
        file_info         =  detection_data.get("file_info", {})
        
        # Extract Analyzed File name from file_info
        original_filename = file_info.get("filename", "Unknown")
        
        # Extract values - handle different data formats
        synthetic_prob    = ensemble_data.get("synthetic_probability", 0) * 100 # Convert to percentage
        authentic_prob    = ensemble_data.get("authentic_probability", 0) * 100 # Convert to percentage
        hybrid_prob       = ensemble_data.get("hybrid_probability", 0) * 100    # Convert to percentage
        confidence        = ensemble_data.get("overall_confidence", 0) * 100    # Convert to percentage
        uncertainty       = ensemble_data.get("uncertainty_score", 0) * 100     # Convert to percentage
        consensus         = ensemble_data.get("consensus_level", 0) * 100       # Convert to percentage
        final_verdict     = ensemble_data.get("final_verdict", "Unknown")
        total_time        = performance_data.get("total_time", 0)
        
        # Determine colors based on verdict
        if ("Authentically-Written".lower() in final_verdict.lower()):
            verdict_color = SUCCESS_COLOR

        elif ("Synthetically-Generated".lower() in final_verdict.lower()):
            verdict_color = DANGER_COLOR
        
        elif ("Hybrid".lower() in final_verdict.lower()):
            verdict_color = WARNING_COLOR
        
        else:
            verdict_color = PRIMARY_COLOR
        
        # PAGE 1: Analyzed File, Verdict, Reasoning, Key Indicators 
        # Header
        header_style = ParagraphStyle('HeaderStyle',
                                      parent     = styles['Normal'],
                                      fontName   = 'Helvetica-Bold',
                                      fontSize   = 10,
                                      textColor  = GRAY_DARK,
                                      alignment  = TA_RIGHT,
                                     )
        
        elements.append(Paragraph("TEXT AUTHENTICATION ANALYTICS", header_style))

        elements.append(HRFlowable(width      = "100%", 
                                   thickness  = 1, 
                                   color      = PRIMARY_COLOR, 
                                   spaceAfter = 15,
                                  )
                       )
        
        # Title and main sections
        elements.append(Paragraph("Text Authentication Analysis Report", title_style))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
        
        # Add original filename
        elements.append(Paragraph(f"Analyzed File: {original_filename}", filename_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Add decorative line
        elements.append(HRFlowable(width       = "80%", 
                                   thickness   = 2, 
                                   color       = PRIMARY_COLOR, 
                                   spaceBefore = 10, 
                                   spaceAfter  = 25, 
                                   hAlign      = 'CENTER',
                                  )
                       )
        
        # Quick Stats Banner
        stats_data  = [['Classification', 'Synthetic', 'Authentic', 'Hybrid'],
                       ['Probability', f"{synthetic_prob:.1f}%", f"{authentic_prob:.1f}%", f"{hybrid_prob:.1f}%"]
                      ]
        
        stats_table = Table(stats_data, colWidths = [1.5*inch, 1*inch, 1*inch, 1*inch])

        stats_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
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
                                       ])
                            )

        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Main Verdict Section
        elements.append(Paragraph("DETECTION VERDICT", section_style))
        
        verdict_box_data = [[Paragraph(f"<font size=10 color='{verdict_color}'><b>{final_verdict.upper()}</b></font>", ParagraphStyle('VerdictText', alignment=TA_CENTER)),
                             Paragraph(f"<font size=12>Confidence: <b>{confidence:.1f}%</b></font><br/>" 
                                       f"<font size=10>Uncertainty: {uncertainty:.1f}% | Consensus: {consensus:.1f}%</font>", 
                                       ParagraphStyle('VerdictDetails', alignment=TA_CENTER))
                           ]]
        
        verdict_box      = Table(verdict_box_data, colWidths = [2.5*inch, 3*inch])

        verdict_box.setStyle(TableStyle([('BACKGROUND', (0, 0), (0, 0), GRAY_LIGHT),
                                         ('BACKGROUND', (1, 0), (1, 0), GRAY_LIGHT),
                                         ('BOX', (0, 0), (-1, -1), 1, verdict_color),
                                         ('ROUNDEDCORNERS', [10, 10, 10, 10]),
                                         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                         ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
                                         ('TOPPADDING', (0, 0), (-1, -1), 15),
                                       ]) 
                            )

        elements.append(verdict_box)
        elements.append(Spacer(1, 0.3*inch))
        
        # DETECTION REASONING
        elements.append(Paragraph("DETECTION REASONING", section_style))
        
        # Process summary text and convert to bullet points
        summary_text  = reasoning.summary if hasattr(reasoning, 'summary') else "No reasoning summary available."
        
        # Fix extra spaces first
        summary_text  = ' '.join(summary_text.split())
        
        # Convert **bold** markers to HTML bold tags
        summary_text  = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', summary_text)
        
        # Split into sentences and create bullet points
        sentences     = re.split(r'(?<=[.!?])\s+', summary_text)

        # Create bullet points
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Add bullet point
                elements.append(Paragraph(f"<font color='{PRIMARY_COLOR}'>•</font> {sentence.strip()}", bullet_style))
                
                # Add extra spacing after each bullet point (except the last one)
                if (i < len(sentences) - 1):
                    # Add spacing between bullet points
                    elements.append(Spacer(1, 0.08*inch))  
        
        # KEY INDICATORS 
        if ((hasattr(reasoning, 'key_indicators')) and reasoning.key_indicators and (len(reasoning.key_indicators) > 0)):
            elements.append(Paragraph("KEY INDICATORS", key_indicators_style))
            
            for indicator in reasoning.key_indicators:
                if isinstance(indicator, str):
                    # Fix extra spaces
                    indicator           = ' '.join(indicator.split())

                    # Convert **bold** markers to proper HTML bold tags
                    formatted_indicator = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', indicator)
                    
                    # Fix underscores in metric names
                    formatted_indicator = formatted_indicator.replace('_', ' ')
                    
                    elements.append(Paragraph(f"<font color='{SUCCESS_COLOR}'>•</font> {formatted_indicator}", body_style))
                    elements.append(Spacer(1, 0.05*inch))
        
        elements.append(PageBreak())
        
        # PAGE 2: Content Analysis & Metric Contributions 
        # CONTENT ANALYSIS
        elements.append(Paragraph("CONTENT ANALYSIS", section_style))
        
        domain            = analysis_data.get("domain", "general").replace('_', ' ').upper()
        
        # Convert to percentage
        domain_confidence = analysis_data.get("domain_confidence", 0) * 100 
        text_length       = analysis_data.get("text_length", 0)
        sentence_count    = analysis_data.get("sentence_count", 0)
        
        # Create two-column layout for content analysis 
        content_data      = [[Paragraph("<b>Content Domain</b>", bold_style), Paragraph(f"<font color='{INFO_COLOR}'><b>{domain}</b></font> ({domain_confidence:.1f}% confidence)", body_style)],
                             [Paragraph("<b>Text Statistics</b>", bold_style), Paragraph(f"{text_length:,} words | {sentence_count:,} sentences", body_style)],
                             [Paragraph("<b>Processing Time</b>", bold_style), Paragraph(f"{total_time:.2f} seconds", body_style)],
                             [Paragraph("<b>Analysis Method</b>", bold_style), Paragraph("Confidence-Weighted Ensemble Aggregation", body_style)],
                            ]
        
        content_table     = Table(content_data, colWidths = [2*inch, 4.5*inch])

        content_table.setStyle(TableStyle([('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                           ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                                           ('FONTSIZE', (0, 0), (-1, -1), 11),       
                                           ('BOTTOMPADDING', (0, 0), (-1, -1), 10),  
                                           ('TOPPADDING', (0, 0), (-1, -1), 10),
                                           ('GRID', (0, 0), (-1, -1), 0.25, GRAY_MEDIUM),
                                           ('BACKGROUND', (0, 0), (0, -1), GRAY_LIGHT),
                                         ])
                              )

        elements.append(content_table)
        elements.append(Spacer(1, 0.4*inch))
        
        # METRIC CONTRIBUTIONS
        elements.append(Paragraph("METRIC CONTRIBUTIONS", section_style))
        
        metric_contributions = ensemble_data.get("metric_contributions", {})

        if (metric_contributions and (len(metric_contributions) > 0)):
            # Create clean table with updated headers
            weight_data = [['METRIC NAME', 'ENSEMBLE WEIGHT (%)']]  
            
            for metric_name, contribution in metric_contributions.items():
                weight       = contribution.get("weight", 0) * 100
                display_name = metric_name.replace('_', ' ').title()

                weight_data.append([Paragraph(display_name, bold_style), Paragraph(f"{weight:.1f}%", body_style)])
            
            # Setup Table Columns
            weight_table = Table(weight_data, colWidths = [4*inch, 2.5*inch])

            weight_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
                                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                              ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                              ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                                              ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                              ('FONTSIZE', (0, 0), (-1, -1), 11),  
                                              ('BOTTOMPADDING', (0, 0), (-1, -1), 10), 
                                              ('TOPPADDING', (0, 0), (-1, -1), 10),
                                              ('GRID', (0, 0), (-1, -1), 0.5, GRAY_MEDIUM),
                                              ('BACKGROUND', (1, 1), (1, -1), GRAY_LIGHT),
                                            ])
                                 )

            elements.append(weight_table)
        
        # Add some filler content to reduce white space
        elements.append(Spacer(1, 0.4*inch))
        elements.append(HRFlowable(width = "100%", thickness = 1, color = PRIMARY_COLOR, spaceBefore = 10, spaceAfter = 10))
        elements.append(Paragraph("<i>Report continues with detailed metric analysis on the following pages...</i>", 
                        ParagraphStyle('ContinueStyle', parent = body_style, fontSize = 10, textColor = GRAY_DARK, alignment = TA_CENTER)))
        
        elements.append(PageBreak())
        
        # PAGE 3: STRUCTURAL & ENTROPY 
        elements.append(Paragraph("DETAILED METRIC ANALYSIS", section_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Filter for STRUCTURAL and ENTROPY only
        page3_metrics = [m for m in detailed_metrics if m.name in ['structural', 'entropy']]
        
        for metric in page3_metrics:
            self._add_detailed_metric_section(elements         = elements, 
                                              metric           = metric, 
                                              small_bold_style = small_bold_style, 
                                              small_style      = small_style, 
                                              bold_style       = bold_style,
                                              PRIMARY_COLOR    = PRIMARY_COLOR, 
                                              SUCCESS_COLOR    = SUCCESS_COLOR,
                                              DANGER_COLOR     = DANGER_COLOR, 
                                              WARNING_COLOR    = WARNING_COLOR, 
                                              GRAY_LIGHT       = GRAY_LIGHT,
                                             )
            
            elements.append(Spacer(1, 0.1*inch))
        
            elements.append(HRFlowable(width = "100%", thickness = 0.5, color = GRAY_MEDIUM, spaceBefore = 5, spaceAfter = 15))
        
        elements.append(PageBreak())
        
        # PAGE 4: PERPLEXITY & SEMANTIC ANALYSIS 
        elements.append(Paragraph("DETAILED METRIC ANALYSIS", section_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Filter for PERPLEXITY and SEMANTIC_ANALYSIS only
        page4_metrics = [m for m in detailed_metrics if m.name in ['perplexity', 'semantic_analysis']]
        
        for metric in page4_metrics:
            self._add_detailed_metric_section(elements         = elements, 
                                              metric           = metric, 
                                              small_bold_style = small_bold_style, 
                                              small_style      = small_style, 
                                              bold_style       = bold_style,
                                              PRIMARY_COLOR    = PRIMARY_COLOR, 
                                              SUCCESS_COLOR    = SUCCESS_COLOR, 
                                              DANGER_COLOR     = DANGER_COLOR, 
                                              WARNING_COLOR    = WARNING_COLOR, 
                                              GRAY_LIGHT       = GRAY_LIGHT,
                                             )
            
            elements.append(Spacer(1, 0.3*inch))
            elements.append(HRFlowable(width = "100%", thickness = 0.5, color = GRAY_MEDIUM, spaceBefore = 5, spaceAfter = 15))
        
        elements.append(PageBreak())
        
        # PAGE 5: LINGUISTIC & MULTI PERTURBATION STABILITY 
        elements.append(Paragraph("DETAILED METRIC ANALYSIS", section_style))
        elements.append(Spacer(1, 0.1*inch)) 
        
        # Filter for LINGUISTIC and MULTI_PERTURBATION_STABILITY only
        page5_metrics  = [m for m in detailed_metrics if m.name in ['linguistic', 'multi_perturbation_stability']]
        
        # Create a list to hold all content for Page 5
        page5_elements = list()
        
        for i, metric in enumerate(page5_metrics):
            # Create temporary elements list for this metric
            metric_elements = list()
            
            # Add metric section to temporary list
            self._add_detailed_metric_section(elements         = metric_elements, 
                                              metric           = metric, 
                                              small_bold_style = small_bold_style, 
                                              small_style      = small_style, 
                                              bold_style       = bold_style,
                                              PRIMARY_COLOR    = PRIMARY_COLOR, 
                                              SUCCESS_COLOR    = SUCCESS_COLOR, 
                                              DANGER_COLOR     = DANGER_COLOR, 
                                              WARNING_COLOR    = WARNING_COLOR, 
                                              GRAY_LIGHT       = GRAY_LIGHT,
                                             )
            
            # Add to page5_elements
            page5_elements.extend(metric_elements)
            
            # Add separator if not the last metric
            if (i < len(page5_metrics) - 1):
                page5_elements.append(Spacer(1, 0.05*inch))  # Minimal spacing
                page5_elements.append(HRFlowable(width = "100%", thickness = 0.5, color = GRAY_MEDIUM, spaceBefore = 5, spaceAfter = 10))
        
        # Add all page 5 elements to main elements
        elements.extend(page5_elements)
        
        elements.append(PageBreak())
        
        
        # RECOMMENDATIONS
        if ((hasattr(reasoning, 'recommendations')) and reasoning.recommendations):
            elements.append(Paragraph("RECOMMENDATIONS", section_style))
            elements.append(Spacer(1, 0.1*inch)) 
            
            for i, recommendation in enumerate(reasoning.recommendations):
                # Alternate colors for visual interest
                if (i % 3 == 0):
                    rec_color = SUCCESS_COLOR

                elif (i % 3 == 1):
                    rec_color = INFO_COLOR

                else:
                    rec_color = WARNING_COLOR
                
                # Clean up recommendation text - fix spaces and bold markers
                clean_rec    = ' '.join(recommendation.split())
                clean_rec    = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_rec)
                clean_rec    = clean_rec.replace('_', ' ')
                
                rec_box_data = [[Paragraph(f"<font color='{rec_color}'>✓</font> {clean_rec}", body_style)]]
                rec_box      = Table(rec_box_data, colWidths = [6.5*inch])

                rec_box.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), GRAY_LIGHT),
                                             ('BOX', (0, 0), (-1, -1), 1, rec_color),
                                             ('PADDING', (0, 0), (-1, -1), 10),
                                             ('LEFTPADDING', (0, 0), (-1, -1), 8),
                                             ('BOTTOMMARGIN', (0, 0), (-1, -1), 6),
                                           ])
                                )

                elements.append(rec_box)
                elements.append(Spacer(1, 0.2*inch))
        
        # Footer with watermark
        elements.append(Spacer(1, 0.2*inch))
        elements.append(HRFlowable(width = "100%", thickness = 0.5, color = GRAY_MEDIUM, spaceAfter = 8))
        
        # Extract report ID from filename
        report_id   = filename.replace('.pdf', '')
        
        footer_text = (f"Generated by Text Authenticator v1.0 | "
                       f"Processing Time: {total_time:.2f}s | "
                       f"Report ID: {report_id}")
        
        elements.append(Paragraph(footer_text, footer_style))
        elements.append(Paragraph("Confidential Analysis Report • © 2025 Text Authentication Analytics", 
                        ParagraphStyle('Copyright', parent = footer_style, fontSize = 8, textColor = GRAY_MEDIUM)))
        
        # Build PDF
        doc.build(elements)
        
        logger.info(f"PDF report saved: {output_path}")
        
        return output_path
    
    
    def _add_detailed_metric_section(self, elements, metric, small_bold_style, small_style, bold_style, PRIMARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR, GRAY_LIGHT):
        """
        Add a detailed metric section to the PDF
        """
        # Import needed components
        from reportlab.platypus import Paragraph, Table, Spacer
        from reportlab.platypus import TableStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        
        # Determine metric color based on verdict
        if (metric.verdict == "Authentic Text"):
            metric_color = SUCCESS_COLOR
            prob_color   = SUCCESS_COLOR

        elif (metric.verdict == "Synthetic Text"):
            metric_color = DANGER_COLOR
            prob_color   = DANGER_COLOR

        else:
            metric_color = WARNING_COLOR
            prob_color   = WARNING_COLOR
        
        # Create professional metric header
        metric_display_name = metric.name.replace('_', ' ').upper()
        
        # Metric title and description
        subsection_style    = ParagraphStyle('SubsectionStyle',
                                             parent      = ParagraphStyle('Normal'),
                                             fontName    = 'Helvetica-Bold',
                                             fontSize    = 12,
                                             textColor   = PRIMARY_COLOR,
                                             spaceAfter  = 8,
                                             spaceBefore = 16,
                                             alignment=TA_LEFT,
                                            )
        
        elements.append(Paragraph(f"<b>{metric_display_name}</b>", subsection_style))
        elements.append(Paragraph(f"<i>{metric.description}</i>", small_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Key metrics in a clean table
        key_metrics_data  = [[Paragraph("<b>Verdict</b>", bold_style), Paragraph(f"<font color='{metric_color}'><b>{metric.verdict}</b></font>", bold_style), Paragraph("<b>Weight</b>", bold_style), Paragraph(f"<b>{metric.weight:.1f}%</b>", bold_style)],
                             [Paragraph("<b>Synthetic Probability</b>", bold_style), Paragraph(f"<font color='{prob_color}'><b>{metric.synthetic_probability:.1f}%</b></font>", bold_style), Paragraph("<b>Confidence</b>", bold_style), Paragraph(f"<b>{metric.confidence:.1f}%</b>", bold_style)]
                            ]
        
        key_metrics_table = Table(key_metrics_data, colWidths = [1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])

        key_metrics_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), GRAY_LIGHT),
                                               ('GRID', (0, 0), (-1, -1), 0.5, colors.white),
                                               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                               ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                               ('TOPPADDING', (0, 0), (-1, -1), 8),
                                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                             ])
                                  )
        
        elements.append(key_metrics_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Detailed metrics in a compact table
        if metric.detailed_metrics and len(metric.detailed_metrics) > 0:
            # Create table with all metrics
            detailed_data = list()
            
            # Sort metrics alphabetically
            sorted_items  = sorted(metric.detailed_metrics.items())
            
            # Group into rows with 3 metrics per row
            for i in range(0, len(sorted_items), 3):
                row = []
                # Add up to 3 metrics per row
                for j in range(3):
                    if i + j < len(sorted_items):
                        key, value = sorted_items[i + j]
                        # Format key name properly
                        display_key     = key.replace('_', ' ').title()
                        formatted_value = self._format_metric_value(key, value)
                        row.append(Paragraph(f"<font size=9><b>{display_key}:</b></font>", small_bold_style))
                        row.append(Paragraph(f"<font size=9>{formatted_value}</font>", small_style))
                    
                    else:
                        row.append("")
                        row.append("")
                
                detailed_data.append(row)
            
            if detailed_data:
                # Calculate column widths dynamically
                col_width      = 6.5 * inch / 6  # 6 columns total
                col_widths     = [col_width] * 6
                
                detailed_table = Table(detailed_data, colWidths = col_widths)

                detailed_table.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 8),
                                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                                                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                                                    ('GRID', (0, 0), (-1, -1), 0.2, colors.grey),
                                                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                                    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                                                    ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
                                                    ('ALIGN', (5, 0), (5, -1), 'RIGHT'),
                                                  ])
                                       )

                elements.append(detailed_table)
    

    def _format_metric_value(self, key: str, value: Any) -> str:
        """
        Format metric value based on its type
        """
        if not isinstance(value, (int, float)):
            return str(value)
        
        key_lower = key.lower()
        
        if ('perplexity' in key_lower):
            if (value > 1000):
                return f"{value:,.0f}"

            else:
                return f"{value:.2f}"

        elif (('probability' in key_lower) or ('confidence' in key_lower)):
            return f"{value:.1f}%"

        elif ('entropy' in key_lower):
            return f"{value:.2f}"

        elif (('ratio' in key_lower) or ('score' in key_lower)):
            if (0 <= value <= 1):
                return f"{value:.3f}"

            else:
                return f"{value:.2f}"

        elif (key_lower in ['num_sentences', 'num_words', 'vocabulary_size']):
            return f"{int(value):,}"
        
        elif (('length' in key_lower) or ('size' in key_lower)):
            return f"{value:.2f}"
        
        else:
            return f"{value:.3f}"


# Export
__all__ = ["ReportGenerator"]