# DEPENDENCIES
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from typing import Optional
from config.enums import Domain
from config.settings import settings
from concurrent.futures import Executor
from config.schemas import MetricResult
from config.schemas import EnsembleResult
from metrics.entropy import EntropyMetric
from config.schemas import DetectionResult
from concurrent.futures import as_completed
from metrics.perplexity import PerplexityMetric
from metrics.linguistic import LinguisticMetric
from metrics.structural import StructuralMetric
from concurrent.futures import ThreadPoolExecutor
from config.schemas import LanguageDetectionResult
from processors.text_processor import TextProcessor
from processors.text_processor import ProcessedText
from processors.domain_classifier import DomainClassifier
from processors.domain_classifier import DomainPrediction
from processors.language_detector import LanguageDetector
from services.ensemble_classifier import EnsembleClassifier
from metrics.semantic_analysis import SemanticAnalysisMetric
from metrics.multi_perturbation_stability import MultiPerturbationStabilityMetric


class DetectionOrchestrator:
    """
    Coordinates the entire detection pipeline from text input to final results

    Pipeline:
    1. Text preprocessing
    2. Domain classification
    3. Language detection (optional)
    4. Metric execution (parallel/sequential)
    5. Ensemble aggregation
    6. Result generation
    """
    def __init__(self, enable_language_detection: bool = False, skip_expensive_metrics: bool = False, parallel_executor: Optional[Executor] = None, parallel_execution: bool = True):
        """
        Initialize detection orchestrator
        
        Arguments:
        ----------
            enable_language_detection { bool } : Enable language detection step
            skip_expensive_metrics    { bool } : Skip computationally expensive metrics
            parallel_executor         { Executor } : Thread/Process executor for parallel processing
            parallel_execution        { bool } : Enable parallel metric execution
        """
        self.enable_language_detection = enable_language_detection
        self.skip_expensive_metrics    = skip_expensive_metrics
        self.parallel_executor         = parallel_executor
        self.parallel_execution        = parallel_execution
        
        # Initialize processors
        self.text_processor            = TextProcessor()

        self.domain_classifier         = DomainClassifier()
        self.language_detector         = LanguageDetector(use_model = True) if self.enable_language_detection else None
        
        # Initialize metrics
        self.metrics                   = self._initialize_metrics()
        
        # Initialize ensemble
        self.ensemble                  = EnsembleClassifier(primary_method       = "confidence_calibrated",
                                                            fallback_method      = "domain_weighted",
                                                            min_metrics_required = 3,
                                                           )
        
        logger.info(f"DetectionOrchestrator initialized (language_detection={enable_language_detection}, "
                    f"skip_expensive={skip_expensive_metrics}, parallel={parallel_execution})")
    

    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize all enabled metrics
        """
        metrics        = dict()
        
        # Define metric initialization order (simpler metrics first)
        metric_classes = [("structural", StructuralMetric),
                          ("entropy", EntropyMetric),
                          ("perplexity", PerplexityMetric),
                          ("semantic_analysis", SemanticAnalysisMetric),
                          ("linguistic", LinguisticMetric),
                          ("multi_perturbation_stability", MultiPerturbationStabilityMetric),
                         ]
        
        for name, metric_class in metric_classes:
            try:
                metrics[name] = metric_class()
                logger.debug(f"{name} metric initialized")
            
            except Exception as e:
                logger.error(f"Failed to initialize {name} metric: {repr(e)}")
        
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
            
            # Initialize processors
            self._initialize_processors()
            
            # Initialize metrics
            successful_metrics = self._initialize_metrics_components()
            
            # Need at least 3 metrics for reliable detection
            pipeline_ready     = (successful_metrics >= 3)
            
            if pipeline_ready:
                logger.success(f"Detection pipeline initialized: {successful_metrics}/{len(self.metrics)} metrics ready")
            
            else:
                logger.warning(f"Pipeline may be unreliable: only {successful_metrics} metrics initialized (need at least 3)")
            
            return pipeline_ready
            
        except Exception as e:
            logger.error(f"Failed to initialize detection pipeline: {repr(e)}")
            return False
    

    def _initialize_processors(self) -> None:
        """
        Initialize processor components
        """
        # Initialize domain classifier
        if not self.domain_classifier.initialize():
            logger.warning("Domain classifier initialization failed")
        
        # Initialize language detector
        if self.language_detector and not self.language_detector.initialize():
            logger.warning("Language detector initialization failed")
    

    def _initialize_metrics_components(self) -> int:
        """
        Initialize metric components and return count of successful initializations
        """
        successful_metrics = 0
        
        for name, metric in self.metrics.items():
            try:
                if metric.initialize():
                    successful_metrics += 1
                    logger.debug(f"✓ {name} metric ready")
                
                else:
                    logger.warning(f"✗ {name} metric initialization failed")
            
            except Exception as e:
                logger.error(f"Error initializing {name} metric: {repr(e)}")
        
        return successful_metrics
    

    def analyze(self, text: str, domain: Optional[Domain] = None, **kwargs) -> DetectionResult:
        """
        Analyze text and detect if synthetically-generated
        
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
            # Step 1: Preprocess text
            processed_text                         = self._preprocess_text(text     = text, 
                                                                           warnings = warnings,
                                                                          )
            
            # Step 2: Detect language
            language_result                        = self._detect_language(processed_text = processed_text, 
                                                                           warnings       = warnings,
                                                                          )
            
            # Step 3: Classify domain
            domain_prediction, domain              = self._classify_domain(processed_text = processed_text, 
                                                                           user_domain    = domain, 
                                                                           warnings       = warnings,
                                                                          )
            
            # Step 4: Execute metrics (parallel or sequential)
            metric_results, metrics_execution_time = self._execute_metrics_parallel(processed_text = processed_text, 
                                                                                    domain         = domain, 
                                                                                    warnings       = warnings, 
                                                                                    errors         = errors,
                                                                                    **kwargs
                                                                                   )
                                                                                
            # Step 5: Ensemble aggregation
            ensemble_result                        = self._aggregate_results(metric_results = metric_results, 
                                                                             domain         = domain, 
                                                                             errors         = errors,
                                                                            )
            
            # Step 6: Compile final result
            processing_time                        = time.time() - start_time
            
            return self._compile_result(ensemble_result         = ensemble_result,
                                        processed_text          = processed_text,
                                        domain_prediction       = domain_prediction,
                                        language_result         = language_result,
                                        metric_results          = metric_results,
                                        processing_time         = processing_time,
                                        metrics_execution_time  = metrics_execution_time,
                                        warnings                = warnings,
                                        errors                  = errors,
                                        **kwargs,
                                       )
            
        except Exception as e:
            logger.error(f"Fatal error in detection pipeline: {repr(e)}")
            return self._create_error_result(text, str(e), start_time)
    

    def _preprocess_text(self, text: str, warnings: List[str]) -> ProcessedText:
        """
        Preprocess text
        """
        logger.info("Step 1: Preprocessing text...")
        processed_text = self.text_processor.process(text = text)
        
        if not processed_text.is_valid:
            logger.warning(f"Text validation failed: {processed_text.validation_errors}")
            warnings.extend(processed_text.validation_errors)
        
        return processed_text
    

    def _detect_language(self, processed_text: ProcessedText, warnings: List[str]) -> Optional[LanguageDetectionResult]:
        """
        Detect language
        """
        if not self.language_detector:
            return None
        
        logger.info("Step 2: Detecting language...")
        
        try:
            language_result = self.language_detector.detect(processed_text.cleaned_text)
            
            # Add relevant warnings
            if (language_result.primary_language.value != "en"):
                warnings.append(f"Non-English text detected ({language_result.primary_language.value}). Detection accuracy may be reduced.")
            
            if language_result.is_multilingual:
                warnings.append("Multilingual content detected")
            
            if (language_result.evidence_strength < 0.7):
                warnings.append(f"Low language detection evidence_strength ({language_result.evidence_strength:.2f})")
            
            return language_result
            
        except Exception as e:
            logger.warning(f"Language detection failed: {repr(e)}")
            warnings.append("Language detection failed")
            return None
    

    def _classify_domain(self, processed_text: ProcessedText, user_domain: Optional[Domain], warnings: List[str]) -> Tuple[DomainPrediction, Domain]:
        """
        Classify domain
        """
        logger.info("Step 3: Classifying domain...")
        
        if user_domain is not None:
            # Use provided domain
            domain_prediction = DomainPrediction(primary_domain    = user_domain,
                                                 secondary_domain  = None,
                                                 evidence_strength = 1.0,
                                                 domain_scores     = {user_domain.value: 1.0},
                                                )
            domain            = user_domain
        
        else:
            # Automatically classify domain
            try:
                domain_prediction = self.domain_classifier.classify(processed_text.cleaned_text)
                domain            = domain_prediction.primary_domain
                
                if (domain_prediction.evidence_strength < 0.5):
                    warnings.append(f"Low domain classification Evidence Strength ({domain_prediction.evidence_strength:.2f})")
            
            except Exception as e:
                logger.warning(f"Domain classification failed: {repr(e)}")
                domain_prediction = DomainPrediction(primary_domain    = Domain.GENERAL,
                                                     secondary_domain  = None,
                                                     evidence_strength = 0.5,
                                                     domain_scores     = {},
                                                    )
                domain            = Domain.GENERAL
                warnings.append("Domain classification failed, using GENERAL")
        
        logger.info(f"Detected domain: {domain.value} (Evidence Strength: {domain_prediction.evidence_strength:.2f})")
        return domain_prediction, domain
    

    def _execute_metrics_parallel(self, processed_text: ProcessedText, domain: Domain, warnings: List[str], errors: List[str], **kwargs) -> Tuple[Dict[str, MetricResult], Dict[str, float]]:
        """
        Execute metrics calculations in parallel with fallback to sequential
        
        Returns:
        --------
            Tuple[Dict[str, MetricResult], Dict[str, float]]: Metric results and execution times
        """
        logger.info("Step 4: Executing detection metrics calculations...")
        
        # Check if we should use parallel execution
        use_parallel = self.parallel_execution and self.parallel_executor is not None
        
        if use_parallel:
            logger.info("Using parallel execution for metrics")
            try:
                return self._execute_metrics_parallel_impl(processed_text = processed_text,
                                                           domain         = domain,
                                                           warnings       = warnings,
                                                           errors         = errors,
                                                           **kwargs
                                                          )

            except Exception as e:
                logger.warning(f"Parallel execution failed, falling back to sequential: {repr(e)}")
                warnings.append(f"Parallel execution failed: {str(e)[:100]}")
                
                return self._execute_metrics_sequential(processed_text = processed_text,
                                                        domain         = domain,
                                                        warnings       = warnings,
                                                        errors         = errors,
                                                        **kwargs
                                                       )
        
        else:
            logger.info("Using sequential execution for metrics")
            return self._execute_metrics_sequential(processed_text = processed_text,
                                                    domain         = domain,
                                                    warnings       = warnings,
                                                    errors         = errors,
                                                    **kwargs
                                                   )
    

    def _execute_metrics_parallel_impl(self, processed_text: ProcessedText, domain: Domain, warnings: List[str], errors: List[str], **kwargs) -> Tuple[Dict[str, MetricResult], Dict[str, float]]:
        """
        Execute metrics in parallel using thread pool
        """
        metric_results         = dict()
        metrics_execution_time = dict()
        futures                = dict()
        
        # Submit all metric computations to thread pool
        for name, metric in self.metrics.items():
            # Skip expensive metrics if configured
            if (self.skip_expensive_metrics and (name == "multi_perturbation_stability")):
                logger.info(f"Skipping expensive metric: {name}")
                continue
            
            # Submit task to thread pool
            future = self.parallel_executor.submit(self._compute_metric_wrapper,
                                                   name           = name,
                                                   metric         = metric,
                                                   text           = processed_text.cleaned_text,
                                                   domain         = domain,
                                                   skip_expensive = self.skip_expensive_metrics,
                                                   warnings       = warnings,
                                                   errors         = errors
                                                  )
            futures[future] = name
        
        # Collect results as they complete
        completed_count = 0
        total_metrics   = len(futures)
        
        for future in as_completed(futures):
            name             = futures[future]
            completed_count += 1
            
            try:
                result, execution_time, metric_warnings = future.result(timeout = 300)  # 5 minute timeout
                
                if result:
                    metric_results[name]         = result
                    metrics_execution_time[name] = execution_time
                    
                    if result.error:
                        warnings.append(f"{name} metric error: {result.error}")
                    
                    if metric_warnings:
                        warnings.extend(metric_warnings)
                    
                    logger.debug(f"Parallel metric completed: {name} ({execution_time:.2f}s) - {completed_count}/{total_metrics}")
                
            except Exception as e:
                logger.error(f"Error computing metric {name} in parallel: {repr(e)}")
                errors.append(f"{name}: {repr(e)}")
                
                # Create error result
                metric_results[name] = MetricResult(metric_name           = name,
                                                    synthetic_probability = 0.5,
                                                    authentic_probability = 0.5,
                                                    hybrid_probability    = 0.0,
                                                    confidence            = 0.0,
                                                    error                 = repr(e),
                                                   )

                metrics_execution_time[name] = 0.0
        
        logger.info(f"Parallel execution completed: {len(metric_results)}/{len(self.metrics)} metrics successful")
        return metric_results, metrics_execution_time
    

    def _compute_metric_wrapper(self, name: str, metric: Any, text: str, domain: Domain, skip_expensive: bool, warnings: List[str], errors: List[str]) -> Tuple[Optional[MetricResult], float, List[str]]:
        """
        Wrapper function for parallel metric computation
        """
        metric_start    = time.time()
        metric_warnings = list()
        
        try:
            logger.debug(f"Computing metric in parallel: {name}")
            
            result = metric.compute(text           = text,
                                    domain         = domain,
                                    skip_expensive = skip_expensive,
                                   )
            
            execution_time = time.time() - metric_start
            
            return result, execution_time, metric_warnings
            
        except Exception as e:
            logger.error(f"Error computing metric {name} in wrapper: {repr(e)}")
            execution_time = time.time() - metric_start
            
            # Create error result
            error_result = MetricResult(metric_name           = name,
                                        synthetic_probability = 0.5,
                                        authentic_probability = 0.5,
                                        hybrid_probability    = 0.0,
                                        confidence            = 0.0,
                                        error                 = repr(e),
                                       )
            
            return error_result, execution_time, metric_warnings
    

    def _execute_metrics_sequential(self, processed_text: ProcessedText, domain: Domain, warnings: List[str], errors: List[str], **kwargs) -> Tuple[Dict[str, MetricResult], Dict[str, float]]:
        """
        Execute metrics calculations sequentially (fallback method)
        """
        metric_results         = dict()
        metrics_execution_time = dict()
        
        for name, metric in self.metrics.items():
            metric_start = time.time()
           
            try:
                # Skip expensive metrics if configured
                if (self.skip_expensive_metrics and (name == "multi_perturbation_stability")):
                    logger.info(f"Skipping expensive metric: {name}")
                    continue
                
                logger.debug(f"Computing metric sequentially: {name}")
                
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
                metric_results[name] = MetricResult(metric_name           = name,
                                                    synthetic_probability = 0.5,
                                                    authentic_probability = 0.5,
                                                    hybrid_probability    = 0.0,
                                                    confidence            = 0.0,
                                                    error                 = repr(e),
                                                   )
            
            finally:
                metrics_execution_time[name] = time.time() - metric_start
        
        logger.info(f"Sequential execution completed: {len(metric_results)} metrics computed")
        return metric_results, metrics_execution_time
    

    def _aggregate_results(self, metric_results: Dict[str, MetricResult], domain: Domain, errors: List[str]) -> EnsembleResult:
        """
        Ensemble aggregation
        """
        logger.info("Step 5: Aggregating results with ensemble...")
        
        try:
            ensemble_result = self.ensemble.predict(metric_results = metric_results,
                                                    domain         = domain,
                                                   )
            
            logger.success(f"Ensemble result: {ensemble_result.final_verdict} (Synthetic probability: {ensemble_result.synthetic_probability:.1%}, confidence: {ensemble_result.overall_confidence:.2f})")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {repr(e)}")
            errors.append(f"Ensemble: {repr(e)}")
            
            # Create fallback result    
            return EnsembleResult(final_verdict         = "Uncertain",
                                  synthetic_probability = 0.5,
                                  authentic_probability = 0.5,
                                  hybrid_probability    = 0.0,
                                  overall_confidence    = 0.0,
                                  domain                = domain,
                                  metric_results        = metric_results,
                                  metric_weights        = {},
                                  weighted_scores       = {},
                                  reasoning             = ["Ensemble aggregation failed"],
                                  uncertainty_score     = 1.0,
                                  consensus_level       = 0.0,
                                 )
    

    def _compile_result(self, ensemble_result: EnsembleResult, processed_text: ProcessedText, domain_prediction: DomainPrediction, language_result: Optional[LanguageDetectionResult],
                        metric_results: Dict[str, MetricResult], processing_time: float, metrics_execution_time: Dict[str, float], warnings: List[str], errors: List[str], **kwargs) -> DetectionResult:
        """
        Compile final detection result
        """
        logger.info("Step 6: Compiling final detection result...")
        
        # Include file info if provided
        file_info      = kwargs.get('file_info')
        
        # Add parallel execution info
        execution_mode = "parallel" if (self.parallel_execution and self.parallel_executor) else "sequential"
        
        return DetectionResult(ensemble_result        = ensemble_result,
                               processed_text         = processed_text,
                               domain_prediction      = domain_prediction,
                               language_result        = language_result,
                               metric_results         = metric_results,
                               processing_time        = processing_time,
                               metrics_execution_time = metrics_execution_time,
                               warnings               = warnings,
                               errors                 = errors,
                               file_info              = file_info,
                               execution_mode         = execution_mode,
                              )
    

    def _create_error_result(self, text: str, error_message: str, start_time: float) -> DetectionResult:
        """
        Create error result when pipeline fails
        """
        processing_time = time.time() - start_time
        
        return DetectionResult(ensemble_result        = EnsembleResult(final_verdict         = "Uncertain",
                                                                       synthetic_probability = 0.5,
                                                                       authentic_probability = 0.5,
                                                                       hybrid_probability    = 0.0,
                                                                       overall_confidence    = 0.0,
                                                                       domain                = Domain.GENERAL,
                                                                       metric_results        = {},
                                                                       metric_weights        = {},
                                                                       weighted_scores       = {},
                                                                       reasoning             = [f"Fatal error: {error_message}"],
                                                                       uncertainty_score     = 1.0,
                                                                       consensus_level       = 0.0,
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
                               domain_prediction      = DomainPrediction(primary_domain    = Domain.GENERAL,
                                                                         secondary_domain  = None,
                                                                         evidence_strength = 0.0,
                                                                         domain_scores     = {},
                                                                        ),
                               language_result        = None,
                               metric_results         = {},
                               processing_time        = processing_time,
                               metrics_execution_time = {},
                               warnings               = [],
                               errors                 = [f"Fatal error: {error_message}"],
                               file_info              = None,
                               execution_mode         = "error",
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
                results.append(self._create_error_result(text, str(e), time.time()))
        
        successful = sum(1 for r in results if r.ensemble_result.final_verdict != "Uncertain")
        logger.info(f"Batch analysis complete: {successful}/{len(texts)} processed successfully")
        
        return results
    

    def cleanup(self):
        """
        Clean up resources
        """
        logger.info("Cleaning up detection orchestrator...")
        
        # Clean up metrics
        self._cleanup_metrics()
        
        # Clean up processors
        self._cleanup_processors()
        
        # Clean up parallel executor if we own it
        if hasattr(self, '_own_executor') and self._own_executor:
            try:
                self.parallel_executor.shutdown(wait=True)
                logger.debug("Cleaned up parallel executor")
            except Exception as e:
                logger.warning(f"Error cleaning up parallel executor: {repr(e)}")
        
        logger.info("Cleanup complete")
    

    def _cleanup_metrics(self) -> None:
        """
        Clean up metric resources
        """
        for name, metric in self.metrics.items():
            try:
                metric.cleanup()
                logger.debug(f"Cleaned up metric: {name}")
            
            except Exception as e:
                logger.warning(f"Error cleaning up metric {name}: {repr(e)}")
    

    def _cleanup_processors(self) -> None:
        """
        Clean up processor resources
        """
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
    

    @classmethod
    def create_with_executor(cls, max_workers: int = 4, **kwargs):
        """
        Factory method to create orchestrator with its own executor
        
        Arguments:
        ----------
            max_workers { int } : Maximum number of parallel workers

            **kwargs            : Additional arguments for DetectionOrchestrator
            
        Returns:
        --------
            { DetectionOrchestrator } : Orchestrator with thread pool executor
        """
        executor                   = ThreadPoolExecutor(max_workers = max_workers)
        orchestrator               = cls(parallel_executor = executor, **kwargs)
        orchestrator._own_executor = True

        return orchestrator



# Export
__all__ = ["DetectionOrchestrator"]