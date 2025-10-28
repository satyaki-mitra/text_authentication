# DEPENDENCIES
import time
from typing import Any
from typing import Dict
from typing import List
from loguru import logger
from typing import Optional
from dataclasses import dataclass
from config.settings import settings
from metrics.entropy import EntropyMetric
from config.threshold_config import Domain
from metrics.base_metric import MetricResult
from detector.ensemble import EnsembleResult
from metrics.detect_gpt import DetectGPTMetric
from metrics.perplexity import PerplexityMetric
from metrics.linguistic import LinguisticMetric
from metrics.structural import StructuralMetric
from detector.ensemble import EnsembleClassifier
from processors.text_processor import TextProcessor
from processors.text_processor import ProcessedText
from processors.domain_classifier import DomainClassifier
from processors.domain_classifier import DomainPrediction
from processors.language_detector import LanguageDetector
from metrics.semantic_analysis import SemanticAnalysisMetric
from processors.language_detector import LanguageDetectionResult



@dataclass
class DetectionResult:
    """
    Complete detection result with all metadata
    """
    # Final results
    ensemble_result        : EnsembleResult
    
    # Input metadata
    processed_text         : ProcessedText
    domain_prediction      : DomainPrediction
    language_result        : Optional[LanguageDetectionResult]
    
    # Metric details
    metric_results         : Dict[str, MetricResult]
    
    # Performance metrics
    processing_time        : float
    metrics_execution_time : Dict[str, float]
    
    # Warnings and errors
    warnings               : List[str]
    errors                 : List[str]
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"prediction"  : {"verdict"           : self.ensemble_result.final_verdict,
                                 "ai_probability"    : round(self.ensemble_result.ai_probability, 4),
                                 "human_probability" : round(self.ensemble_result.human_probability, 4),
                                 "mixed_probability" : round(self.ensemble_result.mixed_probability, 4),
                                 "confidence"        : round(self.ensemble_result.overall_confidence, 4),
                                },
                "analysis"    : {"domain"              : self.domain_prediction.primary_domain.value,
                                 "domain_confidence"   : round(self.domain_prediction.confidence, 4),
                                 "language"            : self.language_result.primary_language.value if self.language_result else "unknown",
                                 "language_confidence" : round(self.language_result.confidence, 4) if self.language_result else 0.0,
                                 "text_length"         : self.processed_text.word_count,
                                 "sentence_count"      : self.processed_text.sentence_count,
                                },
                "metrics"     : {name: result.to_dict() for name, result in self.metric_results.items()},
                "ensemble"    : self.ensemble_result.to_dict(),
                "performance" : {"total_time"   : round(self.processing_time, 3),
                                 "metrics_time" : {name: round(t, 3) for name, t in self.metrics_execution_time.items()},
                                },
                "warnings"    : self.warnings,
                "errors"      : self.errors,
               }



class DetectionOrchestrator:
    """
    Coordinates the entire detection pipeline from text input to final results.

    Pipeline:
    1. Text preprocessing
    2. Domain classification
    3. Language detection (optional)
    4. Metric execution (parallel/sequential)
    5. Ensemble aggregation
    6. Result generation
    """
    
    def __init__(self, enable_language_detection: bool = False, parallel_execution: bool = False, skip_expensive_metrics: bool = False):
        """
        Initialize detection orchestrator
        
        Arguments:
        ----------
            enable_language_detection { bool } : Enable language detection step
            
            parallel_execution        { bool } : Execute metrics in parallel (future feature)
            
            skip_expensive_metrics    { bool } : Skip computationally expensive metrics
        """
        self.enable_language_detection = enable_language_detection
        self.parallel_execution        = parallel_execution
        self.skip_expensive_metrics    = skip_expensive_metrics
        
        # Initialize processors
        self.text_processor            = TextProcessor(min_text_length = settings.MIN_TEXT_LENGTH,
                                                       max_text_length = settings.MAX_TEXT_LENGTH,
                                                      )
        self.domain_classifier         = DomainClassifier()
        
        if self.enable_language_detection:
            self.language_detector = LanguageDetector(use_model = True)
        
        else:
            self.language_detector = None
        
        # Initialize metrics
        self.metrics                   = self._initialize_metrics()
        
        # Initialize ensemble
        self.ensemble                  = EnsembleClassifier(primary_method       = "confidence_calibrated",
                                                            fallback_method      = "domain_weighted",
                                                            use_ml_ensemble      = False,
                                                            min_metrics_required = 3,
                                                           )
        
        logger.info(f"DetectionOrchestrator initialized (language_detection={enable_language_detection}, skip_expensive={skip_expensive_metrics})")
    

    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize all enabled metrics
        """
        metrics = dict()
        
        # Structural metric (statistical analysis)
        try:
            metrics["structural"] = StructuralMetric()
            logger.debug("Structural metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize structural metric: {repr(e)}")
        
        # Entropy metric
        try:
            metrics["entropy"] = EntropyMetric()
            logger.debug("Entropy metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize entropy metric: {repr(e)}")
        
        # Perplexity metric
        try:
            metrics["perplexity"] = PerplexityMetric()
            logger.debug("Perplexity metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize perplexity metric: {repr(e)}")
        
        # Semantic analysis metric
        try:
            metrics["semantic_analysis"] = SemanticAnalysisMetric()
            logger.debug("Semantic analysis metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize semantic analysis metric: {repr(e)}")
        
        # Linguistic metric
        try:
            metrics["linguistic"] = LinguisticMetric()
            logger.debug("Linguistic metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize linguistic metric: {repr(e)}")
        
        # DetectGPT metric (expensive)
        try:
            metrics["detect_gpt"] = DetectGPTMetric()
            logger.debug("DetectGPT metric initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize DetectGPT metric: {repr(e)}")
        
        logger.info(f"Initialized {len(metrics)} metrics: {list(metrics.keys())}")
        return metrics
    

    def initialize(self) -> bool:
        """
        Initialize all components (load models, etc.)
        
        Returns:
        --------
            { bool } : True if successful, False otherwise
        """
        try:
            logger.info("Initializing detection pipeline...")
            
            # Initialize domain classifier
            if not self.domain_classifier.initialize():
                logger.warning("Domain classifier initialization failed")
            
            # Initialize language detector
            if self.language_detector:
                if not self.language_detector.initialize():
                    logger.warning("Language detector initialization failed")
            
            # Initialize metrics
            successful_metrics = 0
            
            for name, metric in self.metrics.items():
                try:
                    if metric.initialize():
                        successful_metrics += 1
                        logger.debug(f"Metric {name} initialized successfully")
                    
                    else:
                        logger.warning(f"Metric {name} initialization failed")
                
                except Exception as e:
                    logger.error(f"Error initializing metric {name}: {repr(e)}")
            
            # Need at least 3 metrics for reliable detection
            logger.success(f"Detection pipeline initialized: {successful_metrics}/{len(self.metrics)} metrics ready")
            return (successful_metrics >= 3)  
            
        except Exception as e:
            logger.error(f"Failed to initialize detection pipeline: {repr(e)}")
            return False
    

    def analyze(self, text: str, domain: Optional[Domain] = None, **kwargs) -> DetectionResult:
        """
        Analyze text and detect if AI-generated
        
        Arguments:
        ----------
            text       { str }   : Input text to analyze
            
            domain   { Domain }  : Override automatic domain detection
            
            **kwargs             : Additional options
            
        Returns:
        --------
            { DetectionResult }  : DetectionResult with complete analysis
        """
        start_time = time.time()
        warnings   = list()
        errors     = list()
        
        try:
            # Preprocess text
            logger.info("Step 1: Preprocessing text...")
            processed_text = self.text_processor.process(text = text)
            
            if not processed_text.is_valid:
                logger.warning(f"Text validation failed: {processed_text.validation_errors}")
                warnings.extend(processed_text.validation_errors)
                # Continue anyway if text is present
            
            # Detect language
            language_result = None

            if self.language_detector:
                logger.info("Step 2: Detecting language...")
                
                try:
                    language_result = self.language_detector.detect(processed_text.cleaned_text)
                    
                    if (language_result.primary_language.value != "en"):
                        warnings.append(f"Non-English text detected ({language_result.primary_language.value}). Detection accuracy may be reduced.")
                    
                    if (language_result.is_multilingual):
                        warnings.append("Multilingual content detected")
                    
                    if (language_result.confidence < 0.7):
                        warnings.append(f"Low language detection confidence ({language_result.confidence:.2f})")

                except Exception as e:
                    logger.warning(f"Language detection failed: {repr(e)}")
                    warnings.append("Language detection failed")
            
            # Classify domain
            logger.info("Step 3: Classifying domain...")
            if domain is None:
                try:
                    domain_prediction = self.domain_classifier.classify(processed_text.cleaned_text)
                    domain            = domain_prediction.primary_domain
                    
                    if (domain_prediction.confidence < 0.5):
                        warnings.append(f"Low domain classification confidence ({domain_prediction.confidence:.2f})")

                except Exception as e:
                    logger.warning(f"Domain classification failed: {repr(e)}")
                    domain_prediction = DomainPrediction(primary_domain   = Domain.GENERAL,
                                                         secondary_domain = None,
                                                         confidence       = 0.5,
                                                         domain_scores    = {},
                                                        )
                    domain            = Domain.GENERAL

                    warnings.append("Domain classification failed, using GENERAL")
            
            else:
                # Use provided domain
                domain_prediction = DomainPrediction(primary_domain   = domain,
                                                     secondary_domain = None,
                                                     confidence       = 1.0,
                                                     domain_scores    = {domain.value: 1.0},
                                                    )
            
            logger.info(f"Detected domain: {domain.value} (confidence: {domain_prediction.confidence:.2f})")
            
            # Execute metrics calculations
            logger.info("Step 4: Executing detection metrics calculations...")
            metric_results         = dict()
            metrics_execution_time = dict()
            
            for name, metric in self.metrics.items():
                metric_start = time.time()
               
                try:
                    # Check if we should skip expensive metrics
                    if (self.skip_expensive_metrics and (name == "detect_gpt")):
                        logger.info(f"Skipping expensive metric: {name}")
                        continue
                    
                    logger.debug(f"Computing metric: {name}")
                    
                    result = metric.compute(text           = processed_text.cleaned_text,
                                            domain         = domain,
                                            skip_expensive = self.skip_expensive_metrics,
                                           )
                    
                    metric_results[name] = result
                    
                    if result.error:
                        warnings.append(f"{name} metric error: {result.error}")
                    
                except Exception as e:
                    logger.error(f"Error computing metric {name}: {repr(e)}")
                    errors.append(f"{name}: {repr(e)}")
                    
                    # Create error result
                    metric_results[name] = MetricResult(metric_name       = name,
                                                        ai_probability    = 0.5,
                                                        human_probability = 0.5,
                                                        mixed_probability = 0.0,
                                                        confidence        = 0.0,
                                                        error             = repr(e),
                                                       )
                finally:
                    metrics_execution_time[name] = time.time() - metric_start
            
            logger.info(f"Executed {len(metric_results)} metrics successfully")
            
            # Ensemble aggregation
            logger.info("Step 5: Aggregating results with ensemble...")
            
            try:
                ensemble_result = self.ensemble.predict(metric_results = metric_results,
                                                        domain         = domain,
                                                       )

            except Exception as e:
                logger.error(f"Ensemble prediction failed: {repr(e)}")
                errors.append(f"Ensemble: {repr(e)}")
               
                # Create fallback result    
                ensemble_result = EnsembleResult(final_verdict      = "Error",
                                                 ai_probability     = 0.5,
                                                 human_probability  = 0.5,
                                                 mixed_probability  = 0.0,
                                                 overall_confidence = 0.0,
                                                 domain             = domain,
                                                 metric_results     = metric_results,
                                                 metric_weights     = {},
                                                 weighted_scores    = {},
                                                 reasoning          = ["Ensemble aggregation failed"],
                                                 uncertainty_score  = 1.0,
                                                 consensus_level    = 0.0,
                                                )
            
            # Calculate total processing time
            processing_time     = time.time() - start_time
            
            logger.success(f"Analysis complete: {ensemble_result.final_verdict} "
                           f"(AI probability: {ensemble_result.ai_probability:.1%}, "
                           f"confidence: {ensemble_result.overall_confidence:.2f}) "
                           f"in {processing_time:.2f}s")
            
            return DetectionResult(ensemble_result        = ensemble_result,
                                   processed_text         = processed_text,
                                   domain_prediction      = domain_prediction,
                                   language_result        = language_result,
                                   metric_results         = metric_results,
                                   processing_time        = processing_time,
                                   metrics_execution_time = metrics_execution_time,
                                   warnings               = warnings,
                                   errors                 = errors,
                                  )
            
        except Exception as e:
            logger.error(f"Fatal error in detection pipeline: {repr(e)}")
            processing_time = time.time() - start_time
            
            # Return error result
            return DetectionResult(ensemble_result        = EnsembleResult(final_verdict      = "Error",
                                                                           ai_probability     = 0.5,
                                                                           human_probability  = 0.5,
                                                                           mixed_probability  = 0.0,
                                                                           overall_confidence = 0.0,
                                                                           domain             = Domain.GENERAL,
                                                                           metric_results     = {},
                                                                           metric_weights     = {},
                                                                           weighted_scores    = {},
                                                                           reasoning          = [f"Fatal error: {str(e)}"],
                                                                           uncertainty_score  = 1.0,
                                                                           consensus_level    = 0.0,
                                                                          ),
                                   processed_text         = ProcessedText(original_text       = text,
                                                                          cleaned_text        = "",
                                                                          sentences           = [],
                                                                          words               = [],
                                                                          paragraphs          = [],
                                                                          char_count          = 0,
                                                                          word_count          = 0,
                                                                          sentence_count      = 0,
                                                                          paragraph_count     = 0,
                                                                          avg_sentence_length = 0.0,
                                                                          avg_word_length     = 0.0,
                                                                          is_valid            = False,
                                                                          validation_errors   = ["Processing failed"],
                                                                          metadata            = {},
                                                                         ),
                                   domain_prediction      = DomainPrediction(primary_domain   = Domain.GENERAL,
                                                                             secondary_domain = None,
                                                                             confidence       = 0.0,
                                                                             domain_scores    = {},
                                                                            ),
                                   language_result        = None,
                                   metric_results         = {},
                                   processing_time        = processing_time,
                                   metrics_execution_time = {},
                                   warnings               = [],
                                   errors                 = [f"Fatal error: {repr(e)}"],
                                  )
    

    def batch_analyze(self, texts: List[str], domain: Optional[Domain] = None) -> List[DetectionResult]:
        """
        Analyze multiple texts
        
        Arguments:
        ----------
            texts    { list }  : List of texts to analyze

            domain  { Domain } : Override automatic domain detection
            
        Returns:
        --------
               { list }        : List of DetectionResult objects
        """
        logger.info(f"Batch analyzing {len(texts)} texts...")

        results = list()
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing text {i+1}/{len(texts)}...")
            try:
                result = self.analyze(text   = text, 
                                      domain = domain,
                                     )

                results.append(result)
            
            except Exception as e:
                logger.error(f"Error analyzing text {i+1}: {repr(e)}")
                # Create error result for this text
                error_result = DetectionResult(ensemble_result        = EnsembleResult(final_verdict      = "Error",
                                                                                       ai_probability     = 0.5,
                                                                                       human_probability  = 0.5,
                                                                                       mixed_probability  = 0.0,
                                                                                       overall_confidence = 0.0,
                                                                                       domain             = Domain.GENERAL,
                                                                                       metric_results     = {},
                                                                                       metric_weights     = {},
                                                                                       weighted_scores    = {},
                                                                                       reasoning          = [f"Analysis failed: {str(e)}"],
                                                                                       uncertainty_score  = 1.0,
                                                                                       consensus_level    = 0.0,
                                                                                      ),
                                               processed_text         = ProcessedText(original_text       = text,
                                                                                      cleaned_text        = "",
                                                                                      sentences           = [],
                                                                                      words               = [],
                                                                                      paragraphs          = [],
                                                                                      char_count          = 0,
                                                                                      word_count          = 0,
                                                                                      sentence_count      = 0,
                                                                                      paragraph_count     = 0,
                                                                                      avg_sentence_length = 0.0,
                                                                                      avg_word_length     = 0.0,
                                                                                      is_valid            = False,
                                                                                      validation_errors   = ["Processing failed"],
                                                                                      metadata            = {},
                                                                                     ),
                                               domain_prediction      = DomainPrediction(primary_domain   = Domain.GENERAL,
                                                                                         secondary_domain = None,
                                                                                         confidence       = 0.0,
                                                                                         domain_scores    = {},
                                                                                        ),
                                               language_result        = None,
                                               metric_results         = {},
                                               processing_time        = 0.0,
                                               metrics_execution_time = {},
                                               warnings               = [],
                                               errors                 = [f"Analysis failed: {repr(e)}"],
                                              )
                results.append(error_result)
        
        logger.info(f"Batch analysis complete: {len(results)}/{len(texts)} processed")
        return results
    

    def cleanup(self):
        """
        Clean up resources
        """
        logger.info("Cleaning up detection orchestrator...")
        
        for name, metric in self.metrics.items():
            try:
                metric.cleanup()
                logger.debug(f"Cleaned up metric: {name}")
            
            except Exception as e:
                logger.warning(f"Error cleaning up metric {name}: {repr(e)}")
        
        if self.domain_classifier:
            try:
                self.domain_classifier.cleanup()
                logger.debug("Cleaned up domain classifier")
           
            except Exception as e:
                logger.warning(f"Error cleaning up domain classifier: {repr(e)}")
        
        if self.language_detector:
            try:
                self.language_detector.cleanup()
                logger.debug("Cleaned up language detector")
        
            except Exception as e:
                logger.warning(f"Error cleaning up language detector: {repr(e)}")
        
        logger.info("Cleanup complete")



# Export
__all__ = ["DetectionResult",
           "DetectionOrchestrator",
          ]