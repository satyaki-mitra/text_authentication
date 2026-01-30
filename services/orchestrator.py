# DEPENDENCIES
import time
import numpy as np
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
from config.constants import orchestration_parameters as params
from metrics.multi_perturbation_stability import MultiPerturbationStabilityMetric


class DetectionOrchestrator:
    """
    Simplified detection orchestrator with sequential execution
    
    Pipeline:
    1. Text preprocessing
    2. Domain classification
    3. Language detection (optional)
    4. Metric execution (sequential)
    5. Ensemble aggregation
    6. Result generation
    """
    def __init__(self, enable_language_detection: bool = False, skip_expensive_metrics: bool = False, parallel_executor: Optional[Executor] = None, parallel_execution: bool = True):
        """
        Initialize detection orchestrator
        
        Arguments:
        ----------
            enable_language_detection { bool }   : Enable language detection step
            
            skip_expensive_metrics    { bool }   : Skip computationally expensive metrics

            parallel_executor       { Executor } : Thread/Process executor for parallel processing
           
            parallel_execution        { bool }   : Enable parallel metric execution
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
        self.ensemble                  = EnsembleClassifier(calibration_temperature = 1.3,
                                                            min_metrics_required    = 3,
                                                            execution_mode          = "sequential",
                                                           )
        
        logger.info(f"DetectionOrchestrator initialized (language_detection={enable_language_detection}, skip_expensive={skip_expensive_metrics})")
    

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
            # Preprocess text
            processed_text            = self._preprocess_text(text     = text, 
                                                              warnings = warnings,
                                                             )
            
            # Detect language
            language_result           = self._detect_language(processed_text = processed_text, 
                                                              warnings       = warnings,
                                                             )
            
            # Classify domain
            domain_prediction, domain = self._classify_domain(processed_text = processed_text, 
                                                              user_domain    = domain, 
                                                              warnings       = warnings,
                                                             )

            # Check if text is too long for single analysis
            word_count                = processed_text.word_count

            if (word_count > params.MAX_SINGLE_ANALYSIS_WORDS):
                logger.info(f"Long text detected ({word_count} words), using windowed analysis")
                warnings.append(f"Long text ({word_count} words) analyzed using sliding window approach")
                
                # Use windowed analysis for long texts
                ensemble_result, metric_results, metrics_execution_time = self._analyze_long_text_windowed(processed_text = processed_text,
                                                                                                           domain         = domain,
                                                                                                           warnings       = warnings,
                                                                                                           errors         = errors,
                                                                                                           **kwargs,
                                                                                                          )
            else:
                # Execute metrics sequentially
                metric_results, metrics_execution_time                  = self._execute_metrics_sequential(processed_text = processed_text, 
                                                                                                           domain         = domain, 
                                                                                                           warnings       = warnings, 
                                                                                                           errors         = errors,
                                                                                                           **kwargs,
                                                                                                          )
                                                                                    
                # Ensemble aggregation
                ensemble_result                                         = self._aggregate_results(metric_results = metric_results, 
                                                                                                  domain         = domain, 
                                                                                                  errors         = errors,
                                                                                                 )
            
            # Compile final result
            processing_time = time.time() - start_time
            
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
    

    def _analyze_long_text_windowed(self, processed_text: ProcessedText, domain: Domain, warnings: List[str], errors: List[str], **kwargs) -> Tuple[EnsembleResult, Dict[str, MetricResult], Dict[str, float]]:
        """
        Simplified windowed analysis for long text
        """
        logger.info("Starting windowed analysis for long text...")
        
        # Split text into overlapping windows
        windows = self._create_text_windows(text = processed_text.cleaned_text)
        
        if not windows:
            logger.warning("Failed to create windows, falling back to truncated analysis")
            warnings.append("Windowing failed, using truncated text analysis")

            # Fallback: analyze first MAX_SINGLE_ANALYSIS_WORDS
            truncated_text                         = ' '.join(processed_text.words[:params.MAX_SINGLE_ANALYSIS_WORDS])
            truncated_processed                    = self.text_processor.process(text = truncated_text)
            
            metric_results, metrics_execution_time = self._execute_metrics_sequential(processed_text = truncated_processed,
                                                                                      domain         = domain,
                                                                                      warnings       = warnings,
                                                                                      errors         = errors,
                                                                                      **kwargs,
                                                                                     )
            
            ensemble_result                        = self._aggregate_results(metric_results = metric_results,
                                                                             domain         = domain,
                                                                             errors         = errors,
                                                                            )
            
            return ensemble_result, metric_results, metrics_execution_time
        
        logger.info(f"Created {len(windows)} overlapping windows")
        
        # Analyze each window
        window_results    = list()
        all_metrics_times = dict()
        
        for i, window_text in enumerate(windows):
            logger.debug(f"Analyzing window {i+1}/{len(windows)}")
            
            try:
                # Process window
                window_processed                    = self.text_processor.process(text = window_text)
                
                # Execute metrics on window
                window_metric_results, window_times = self._execute_metrics_sequential(processed_text = window_processed,
                                                                                       domain         = domain,
                                                                                       warnings       = warnings,
                                                                                       errors         = errors,
                                                                                       **kwargs,
                                                                                      )
                
                # Validate window results - skip if too many metrics failed
                valid_metrics                       = sum(1 for mr in window_metric_results.values() if mr.error is None)

                if (valid_metrics < len(window_metric_results) * params.MIN_VALID_METRICS_RATIO_PER_WINDOW):
                    logger.warning(f"Window {i+1} has too many failed metrics ({valid_metrics}/{len(window_metric_results)}), skipping")
                    continue
                
                # Aggregate window results
                window_ensemble                     = self._aggregate_results(metric_results = window_metric_results,
                                                                              domain         = domain,
                                                                              errors         = errors,
                                                                             )
                
                window_results.append({'ensemble' : window_ensemble,
                                       'metrics'  : window_metric_results,
                                       'times'    : window_times,
                                     })
                
                # Accumulate timing data
                for metric_name, time_val in window_times.items():
                    if metric_name not in all_metrics_times:
                        all_metrics_times[metric_name] = []

                    all_metrics_times[metric_name].append(time_val)
                
            except Exception as e:
                logger.error(f"Error analyzing window {i+1}: {repr(e)}")
                errors.append(f"Window {i+1} analysis failed: {str(e)}")
                continue
        
        if not window_results:
            logger.error("All window analyses failed")
            raise Exception("Windowed analysis failed for all windows")
        
        # Aggregate results across windows
        logger.info(f"Aggregating results from {len(window_results)} windows")
        
        aggregated_ensemble = self._aggregate_window_results(window_results = window_results,
                                                             domain         = domain,
                                                            )
        
        # Aggregate metric results (average across windows)
        aggregated_metrics  = self._aggregate_window_metrics(window_results = window_results)
        
        # Average timing data
        avg_metrics_times   = {metric_name: np.mean(times) for metric_name, times in all_metrics_times.items()}
        
        logger.success(f"Windowed analysis complete: {len(window_results)} windows processed")
        
        return aggregated_ensemble, aggregated_metrics, avg_metrics_times
    

    def _create_text_windows(self, text: str) -> List[str]:
        """
        Create overlapping windows from long text
        """
        words   = text.split()

        if (len(words) <= params.MAX_SINGLE_ANALYSIS_WORDS):
            return [text]

        windows = list()
        start   = 0
        size    = params.WINDOW_SIZE_WORDS
        overlap = params.WINDOW_OVERLAP_WORDS
        step    = (size - overlap)

        while (start < len(words)):
            end   = min((start + size), len(words))
            chunk = words[start:end]

            # Discard weak windows
            if (len(chunk) >= max(params.MIN_WINDOW_WORDS_ABSOLUTE, size // 2)):
                windows.append(" ".join(chunk))

            if (end >= len(words)):
                break

            start += step

        return windows
    

    def _aggregate_window_results(self, window_results: List[Dict], domain: Domain) -> EnsembleResult:
        """
        Simplified robust aggregation of windowed results
        
        Strategy:
        - Low variance (< 0.02): Use mean
        - High variance: Use median with confidence penalty
        - Special case: Stability override for creative domains
        """
        ensembles                = [wr["ensemble"] for wr in window_results]
        N                        = len(ensembles)

        synthetic_probabilities  = np.array([e.synthetic_probability for e in ensembles])
        authentic_probabilities  = np.array([e.authentic_probability for e in ensembles])
        hybrid_probabilities     = np.array([e.hybrid_probability for e in ensembles])
        confidences              = np.array([e.overall_confidence for e in ensembles])

        # Calculate variance
        variance                 = float(np.var(synthetic_probabilities))

        # Strategy 1: Low variance - strong agreement
        if (variance < params.WINDOW_LOW_VARIANCE_THRESHOLD):
            synthetic_probability = np.mean(synthetic_probabilities)
            authentic_probability = np.mean(authentic_probabilities)
            hybrid_probability    = np.mean(hybrid_probabilities)
            confidence            = np.mean(confidences)
            aggregation_method    = "mean"
        
        # Strategy 2: High variance - use weighted median
        else:
            # Confidence-weighted percentiles instead of raw median
            weights               = confidences / confidences.sum()
            
            # Weighted average (more robust than median for outliers)
            synthetic_probability = np.average(synthetic_probabilities, weights = weights)
            authentic_probability = np.average(authentic_probabilities, weights = weights)
            hybrid_probability    = np.average(hybrid_probabilities, weights = weights)
            
            # Confidence penalty
            confidence            = np.mean(confidences) * params.HIGH_VARIANCE_CONFIDENCE_MULTIPLIER  
            aggregation_method    = "weighted_average"

        # Normalize
        total_probability = synthetic_probability + authentic_probability + hybrid_probability
        
        if (total_probability > 0):
            synthetic_probability /= total_probability
            authentic_probability /= total_probability
            hybrid_probability    /= total_probability


        # Special case: Stability override for creative domains and extreme cases
        stability_override = False

        # Only apply to creative domains and for very low stability
        if domain in {Domain.CREATIVE, Domain.BLOG_PERSONAL}:
            stability_scores = self._extract_stability_scores(ensembles = ensembles)
            
            if (stability_scores and (np.mean(stability_scores) < params.STABILITY_HARD_OVERRIDE)):
                logger.warning(f"Stability override triggered (mean={np.mean(stability_scores):.4f})")

                # Only boost if MOST windows agree it's synthetic
                synthetic_windows = sum(1 for e in ensembles if e.synthetic_probability > 0.5)

                # Require 60% of windows to agree before override
                if (synthetic_windows / len(ensembles) >= 0.6):
                    logger.warning(f"Stability override triggered (mean={np.mean(stability_scores):.4f}, agreement={synthetic_windows}/{len(ensembles)})")

                    # Less aggressive boost (was 0.75, now 0.65)
                    synthetic_probability = max(synthetic_probability, params.STABILITY_HARD_MIN_SYNTHETIC)
                    total                 = synthetic_probability + authentic_probability

                    if (total > 0):
                        authentic_probability = authentic_probability * (1.0 - synthetic_probability) / (total - synthetic_probability)
                        hybrid_probability    = 0.0
                    
                    # Confidence boost
                    confidence         = min(params.STABILITY_HARD_CONFIDENCE_CAP, confidence + params.STABILITY_HARD_CONFIDENCE_BOOST)
                    stability_override = True


        # Verdict determination 
        margin = abs(synthetic_probability - authentic_probability)
        
        # Lower thresholds to avoid abstention
        if (margin > params.WINDOW_VERDICT_MARGIN) and (confidence > params.WINDOW_VERDICT_CONFIDENCE_GATE):
            verdict = "Synthetically-Generated" if (synthetic_probability > authentic_probability) else "Authentically-Written"
        
        else:
            # Still make a decision if one class clearly dominates
            if (max(synthetic_probability, authentic_probability) > 0.55):
                verdict = "Synthetically-Generated" if (synthetic_probability > authentic_probability) else "Authentically-Written"
            
            else:
                verdict = "Uncertain"

        
        # Generate reasoning
        reasoning = [f"## Windowed Analysis Summary",
                     f"Windows analyzed: {N}",
                     f"Variance: {variance:.3f} (method: {aggregation_method})",
                     f"Stability override: {'Yes' if stability_override else 'No'}",
                     f"Final verdict: {verdict}",
                     f"Confidence: {confidence:.2%}",
                     f"Margin: {margin:.3f}",
                     ""
                    ]
        
        # Show first 5 windows
        for i, e in enumerate(ensembles[:5]):  
            reasoning.append(f"Window {i+1}: {e.final_verdict} (syn={e.synthetic_probability:.2f}, conf={e.overall_confidence:.2f})")

        return EnsembleResult(final_verdict         = verdict,
                              synthetic_probability = synthetic_probability,
                              authentic_probability = authentic_probability,
                              hybrid_probability    = hybrid_probability,
                              overall_confidence    = confidence,
                              domain                = domain,
                              metric_results        = ensembles[0].metric_results,
                              metric_weights        = ensembles[0].metric_weights,
                              weighted_scores       = ensembles[0].weighted_scores,
                              reasoning             = reasoning,
                              uncertainty_score     = variance,
                              consensus_level       = 1.0 - min(1.0, variance * params.WINDOW_VARIANCE_CONSENSUS_SCALE),
                              execution_mode        = f"windowed_{N}_windows",
                             )
                    

    def _extract_stability_scores(self, ensembles: List[EnsembleResult]) -> List[float]:
        """
        Extract stability scores from ensemble results
        """
        stability_scores = list()

        for e in ensembles:
            m = e.metric_results.get("multi_perturbation_stability")
            
            if (m and m.details and ("stability_score" in m.details)):
                stability_scores.append(m.details["stability_score"])
        
        return stability_scores
    

    def _aggregate_window_metrics(self, window_results: List[Dict]) -> Dict[str, MetricResult]:
        """
        Aggregate individual metric results across windows using robust median approach
        """
        if not window_results:
            return {}
        
        # Get metric names from first window
        metric_names       = list(window_results[0]['metrics'].keys())
        aggregated_metrics = dict()
        
        for metric_name in metric_names:
            # Extract this metric from all windows
            metric_results = [wr['metrics'][metric_name] for wr in window_results]
            
            # Filter out failed metrics
            valid_results  = [mr for mr in metric_results if mr.error is None]
            
            if not valid_results:
                # All metrics failed for this type
                aggregated_metrics[metric_name] = metric_results[0]  
                # Return first (with error)
                continue
            
            # Extract probabilities and confidences
            synthetic_probabilities = np.array([mr.synthetic_probability for mr in valid_results])
            authentic_probabilities = np.array([mr.authentic_probability for mr in valid_results])
            hybrid_probabilities    = np.array([mr.hybrid_probability for mr in valid_results])
            confidences             = np.array([mr.confidence for mr in valid_results])
            
            # Use median for robustness
            synthetic_probability   = np.median(synthetic_probabilities)
            authentic_probability   = np.median(authentic_probabilities)
            hybrid_probability      = np.median(hybrid_probabilities)
            confidence              = np.median(confidences)
            
            # Normalize
            total_probability       = synthetic_probability + authentic_probability + hybrid_probability
            
            if (total_probability > 0):
                synthetic_probability /= total_probability
                authentic_probability /= total_probability
                hybrid_probability    /= total_probability
            
            # Create aggregated metric result
            aggregated_metrics[metric_name] = MetricResult(metric_name           = metric_name,
                                                           synthetic_probability = synthetic_probability,
                                                           authentic_probability = authentic_probability,
                                                           hybrid_probability    = hybrid_probability,
                                                           confidence            = confidence,
                                                           details               = {'aggregated_from_windows' : len(window_results),
                                                                                    'valid_windows'           : len(valid_results),
                                                                                    'variance'                : float(np.var(synthetic_probabilities)),
                                                                                   },
                                                           error                 = None,
                                                          )
        
        return aggregated_metrics
    

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
                warnings.append(f"Low language detection confidence ({language_result.evidence_strength:.2f})")
            
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
                domain_text       = " ".join(processed_text.words[:params.MAX_WORDS_FOR_CLASSIFICATION])
                domain_prediction = self.domain_classifier.classify(domain_text)
                domain            = domain_prediction.primary_domain
                
                if (domain_prediction.evidence_strength < 0.5):
                    warnings.append(f"Low domain classification confidence ({domain_prediction.evidence_strength:.2f})")
            
            except Exception as e:
                logger.warning(f"Domain classification failed: {repr(e)}")
                domain_prediction = DomainPrediction(primary_domain    = Domain.GENERAL,
                                                     secondary_domain  = None,
                                                     evidence_strength = 0.5,
                                                     domain_scores     = {},
                                                    )

                domain            = Domain.GENERAL

                warnings.append("Domain classification failed, using GENERAL")
        
        logger.info(f"Detected domain: {domain.value} (confidence: {domain_prediction.evidence_strength:.2f})")
        return domain_prediction, domain
    

    def _execute_metrics_sequential(self, processed_text: ProcessedText, domain: Domain, warnings: List[str], errors: List[str], **kwargs) -> Tuple[Dict[str, MetricResult], Dict[str, float]]:
        """
        Execute metrics calculations sequentially
        """
        logger.info("Step 4: Executing detection metrics calculations (sequential)...")
        
        metric_results         = dict()
        metrics_execution_time = dict()
        
        for name, metric in self.metrics.items():
            metric_start = time.time()
           
            try:
                # Skip expensive metrics if configured
                if (self.skip_expensive_metrics and (name == "multi_perturbation_stability")):
                    logger.info(f"Skipping expensive metric: {name}")
                    continue
                
                logger.debug(f"Computing metric: {name}")
                
                result               = metric.compute(text           = processed_text.cleaned_text,
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
            
            logger.success(f"Ensemble result: {ensemble_result.final_verdict} (Synthetic prob: {ensemble_result.synthetic_probability:.1%}, confidence: {ensemble_result.overall_confidence:.2f})")
            
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
                                  execution_mode        = "sequential",
                                 )
    

    def _compile_result(self, ensemble_result: EnsembleResult, processed_text: ProcessedText, domain_prediction: DomainPrediction, 
                        language_result: Optional[LanguageDetectionResult], metric_results: Dict[str, MetricResult], processing_time: float, 
                        metrics_execution_time: Dict[str, float], warnings: List[str], errors: List[str], **kwargs) -> DetectionResult:
        """
        Compile final detection result
        """
        logger.info("Step 6: Compiling final detection result...")
        
        # Include file info if provided
        file_info      = kwargs.get('file_info')
        
        # Execution mode
        execution_mode = "sequential"

        if ("windowed" in ensemble_result.execution_mode):
            execution_mode = f"{ensemble_result.execution_mode}_sequential"
        
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
                                                                       execution_mode        = "error",
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