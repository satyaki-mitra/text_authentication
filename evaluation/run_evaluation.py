# Dependencies
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Any
from typing import List
from typing import Dict
from scipy import stats
from pathlib import Path
from typing import Tuple
from datetime import datetime
from dataclasses import asdict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix     
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.enums import Domain
from services.orchestrator import DetectionOrchestrator



@dataclass
class EvaluationSample:
    """
    Single evaluation sample
    """
    text_id      : str
    domain       : str
    ground_truth : str  # "human" or "ai"
    text         : str
    file_path    : str
    subset       : str  # "clean", "paraphrased", "cross_model"


@dataclass
class SingleEvalResult:
    """
    Result for a single sample
    """
    text_id         : str
    domain          : str
    ground_truth    : str  # "human" or "ai"
    prediction      : str  # "human", "ai", "hybrid", or "uncertain"
    verdict         : str  # Raw verdict from system: "Synthetically-Generated", "Authentically-Written", "Hybrid", "Uncertain"
    synthetic_prob  : float
    authentic_prob  : float
    hybrid_prob     : float
    confidence      : float
    uncertainty     : float
    processing_time : float
    is_correct      : bool | None
    subset          : str
    word_count      : int  # For length analysis


@dataclass
class AggregatedMetrics:
    """
    Aggregated performance metrics for 4-class system
    """
    # Binary metrics (AI vs Human on decisive predictions)
    precision               : float
    recall                  : float
    f1                      : float
    accuracy                : float
    
    # Coverage metrics
    coverage                : float  # % samples with decisive prediction (not uncertain)
    accuracy_at_cov         : float  # Accuracy on non-uncertain predictions
    f1_at_cov               : float  # F1 on non-uncertain predictions
    
    # Probability-based metrics
    auroc                   : float
    auprc                   : float
    ece                     : float  # Expected Calibration Error
    
    # 4-class specific metrics
    hybrid_detection_rate   : float  # % of AI samples classified as Hybrid
    abstention_rate         : float  # % classified as Uncertain
    
    # Confusion metrics
    confusion_matrix        : List[List[int]]  # 2x2 for decisive predictions only
    support                 : Dict[str, int]
    
    # 4-class breakdown
    verdict_distribution    : Dict[str, int]  # Count of each verdict type


@dataclass
class LengthBucketMetrics:
    """
    Performance metrics for a specific text length range
    """
    min_words       : int
    max_words       : int
    label           : str
    sample_count    : int
    precision       : float
    recall          : float
    f1              : float
    accuracy        : float
    mean_confidence : float
    fp_rate         : float
    fn_rate         : float
    avg_proc_time   : float
    abstention_rate : float


class TextAuthEvaluator:
    """
    Comprehensive evaluation framework for TEXT-AUTH (4-class system)
    
    Handles verdicts:
    - "Synthetically-Generated" → prediction = "ai"
    - "Authentically-Written" → prediction = "human"
    - "Hybrid" → prediction = "hybrid"
    - "Uncertain" → prediction = "uncertain"
    """
    def __init__(self, dataset_path: str = "evaluation", output_dir: str = "evaluation/results"):
        """
        Initialize evaluator
        
        Arguments:
        ----------
            dataset_path { str } : Path to evaluation directory

            output_dir   { str } : Directory to save results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(exist_ok = True, 
                              parents  = True,
                             )
        
        # Initialize orchestrator
        print("\nInitializing TEXT-AUTH Detection Orchestrator...")
        self.orchestrator = DetectionOrchestrator.create_with_executor(max_workers               = 4,
                                                                       enable_language_detection = False,
                                                                       parallel_execution        = True,
                                                                       skip_expensive_metrics    = False,
                                                                      )
        
        if not self.orchestrator.initialize():
            raise RuntimeError("Failed to initialize detection orchestrator")
        
        print("\nOrchestrator initialized successfully\n")
        
        # Storage for results
        self.results  = list()
        self.metadata = dict()
        
        # Load metadata if available
        self._load_metadata()
    

    def _load_metadata(self):
        """
        Load dataset metadata
        """
        metadata_path = self.dataset_path / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            print(f"\nDataset: {self.metadata.get('dataset_name', 'Unknown')}")
            print(f"   Version: {self.metadata.get('version', 'Unknown')}")
            print(f"   Total samples: {self.metadata.get('total_samples', 'Unknown')}")
            print(f"   Human: {self.metadata.get('human_samples', 'Unknown')}")
            print(f"   AI: {self.metadata.get('ai_samples', 'Unknown')}")

            if ('challenge_samples' in self.metadata):
                challenges = self.metadata['challenge_samples']
                print(f"   Paraphrased: {challenges.get('paraphrased', 0)}")
                print(f"   Cross-model: {challenges.get('cross_model', 0)}")

            print()

        else:
            print("\nNo metadata.json found - run create_metadata.py first\n")
    

    def load_dataset(self, domains: List[str] = None, max_samples_per_domain: int = None, subset_filter: str = None) -> List[EvaluationSample]:
        """
        Load evaluation dataset
        
        Arguments:
        ----------
            domains                { list } : List of domains to evaluate (None = all)

            max_samples_per_domain { int }  : Limit samples per domain
            
            subset_filter          { str }  : Only load specific subset
            
        Returns:
        --------
                      { list }              : List of EvaluationSample objects
        """
        samples = list()
        
        # Load clean samples (human + ai)
        if (subset_filter is None or (subset_filter == "clean")):
            for subset_name, subset_dir in [("human", "human"), ("ai", "ai_generated")]:
                subset_path = self.dataset_path / subset_dir
                
                if not subset_path.exists():
                    print(f"Directory not found: {subset_path}")
                    continue
                
                for domain_dir in subset_path.iterdir():
                    if not domain_dir.is_dir():
                        continue
                    
                    domain = domain_dir.name
                    if domains and domain not in domains:
                        continue
                    
                    files = list(domain_dir.glob("*.txt"))
                    if max_samples_per_domain:
                        files = files[:max_samples_per_domain]
                    
                    for file_path in files:
                        try:
                            with open(file_path, 'r', encoding = 'utf-8') as f:
                                text = f.read()
                            
                            samples.append(EvaluationSample(text_id      = file_path.stem,
                                                            domain       = domain,
                                                            ground_truth = subset_name,
                                                            text         = text,
                                                            file_path    = str(file_path),
                                                            subset       = "clean",
                                                           )
                                          )

                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        # Load challenge sets (adversarial)
        if subset_filter is None or subset_filter in ["paraphrased", "cross_model"]:
            adversarial_path = self.dataset_path / "adversarial"
            
            if adversarial_path.exists():
                for challenge_name in ["paraphrased", "cross_model"]:
                    if subset_filter and subset_filter != challenge_name:
                        continue
                    
                    challenge_path = adversarial_path / challenge_name
                    
                    if not challenge_path.exists():
                        continue
                    
                    files = list(challenge_path.glob("*.txt"))
                    
                    for file_path in files:
                        try:
                            with open(file_path, 'r', encoding = 'utf-8') as f:
                                text = f.read()
                            
                            # Extract domain from filename
                            domain = "general"
                            for possible_domain in ["academic", "creative", "ai_ml", "software_dev", "technical_doc", "engineering", "science", "business", "legal", "medical", "journalism", "marketing", "social_media", "blog_personal", "tutorial", "general"]:
                                if possible_domain in file_path.stem:
                                    domain = possible_domain
                                    break
                            
                            # Skip if domain filter active and doesn't match
                            if domains and domain not in domains:
                                continue
                            
                            samples.append(EvaluationSample(text_id      = file_path.stem,
                                                            domain       = domain,
                                                            ground_truth = "ai",  # All adversarial are AI-generated
                                                            text         = text,
                                                            file_path    = str(file_path),
                                                            subset       = challenge_name,
                                                           )
                                          )
                        
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        print(f"\nLoaded {len(samples)} samples")
        
        return samples
    

    def _map_verdict_to_prediction(self, verdict: str) -> str:
        """
        Map system verdict to evaluation prediction class
        
        Arguments:
        ----------
            verdict { str } : Raw verdict from system
            
        Returns:
        --------
            { str } : Mapped prediction ("human", "ai", "hybrid", "uncertain")
        """
        verdict_lower = verdict.lower()
        
        if (("synthetic" in verdict_lower) or ("generated" in verdict_lower)):
            return "ai"
        
        elif (("authentic" in verdict_lower) or ("written" in verdict_lower)):
            return "human"
        
        elif ("hybrid" in verdict_lower):
            return "hybrid"
        
        else:  # "Uncertain" or any other
            return "uncertain"
    

    def run_evaluation(self, samples: List[EvaluationSample]):
        """
        Run evaluation on all samples
        
        Arguments:
        ----------
            samples { list } : List of EvaluationSample objects
        """
        print(f"\nEvaluating {len(samples)} samples...")
        print("=" * 70)
        
        for i, sample in enumerate(tqdm(samples, desc = "Processing")):
            try:
                start_time      = time.time()
                
                # Run detection
                result          = self.orchestrator.analyze(text = sample.text)
                
                proc_time       = time.time() - start_time
                
                # Extract results
                ensemble        = result.ensemble_result
                verdict         = ensemble.final_verdict
                prediction      = self._map_verdict_to_prediction(verdict)
                synthetic_prob  = ensemble.synthetic_probability
                authentic_prob  = ensemble.authentic_probability
                hybrid_prob     = ensemble.hybrid_probability
                confidence      = ensemble.overall_confidence
                uncertainty     = ensemble.uncertainty_score
                word_count      = len(sample.text.split())
                
                # Determine correctness (only for decisive predictions)
                is_correct      = None
                
                if ((prediction == "ai") and (sample.ground_truth == "ai")):
                    is_correct = True
                
                elif ((prediction == "human") and (sample.ground_truth == "human")):
                    is_correct = True
                
                elif ((prediction in ["ai", "human"]) and (prediction != sample.ground_truth)):
                    is_correct = False
                
                # Hybrid is considered correct if ground truth is AI (it detected synthetic content)
                elif ((prediction == "hybrid") and (sample.ground_truth == "ai")):
                    is_correct = True
                
                elif ((prediction == "hybrid") and (sample.ground_truth == "human")):
                    is_correct = False
                
                # Uncertain predictions are neither correct nor incorrect: they are abstentions and handled separately
                
                # Store result
                eval_result = SingleEvalResult(text_id         = sample.text_id,
                                               domain          = sample.domain,
                                               ground_truth    = sample.ground_truth,
                                               prediction      = prediction,
                                               verdict         = verdict,
                                               synthetic_prob  = synthetic_prob,
                                               authentic_prob  = authentic_prob,
                                               hybrid_prob     = hybrid_prob,
                                               confidence      = confidence,
                                               uncertainty     = uncertainty,
                                               processing_time = proc_time,
                                               is_correct      = is_correct,
                                               subset          = sample.subset,
                                               word_count      = word_count,
                                              )
                
                self.results.append(eval_result)
            
            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue
        
        print("\n" + "=" * 70)
        print(f"Evaluation complete: {len(self.results)}/{len(samples)} samples processed")
    

    def calculate_metrics(self, domain: str = None, subset: str = None) -> AggregatedMetrics:
        """
        Calculate aggregated metrics for 4-class system
        
        Arguments:
        ----------
            domain { str }        : Calculate for specific domain only

            subset { str }        : Calculate for specific subset only
            
        Returns:
        --------
            { AggregatedMetrics } : Aggregated metrics
        """
        # Filter results
        filtered = self.results
        
        if domain:
            filtered = [r for r in filtered if r.domain == domain]
        
        if subset:
            filtered = [r for r in filtered if r.subset == subset]
        
        if not filtered:
            return None
        
        # Separate decisive vs uncertain
        decisive     = [r for r in filtered if r.prediction != "uncertain"]
        uncertain    = [r for r in filtered if r.prediction == "uncertain"]
        
        # Calculate coverage
        coverage     = len(decisive) / len(filtered) if filtered else 0.0
        
        # Verdict distribution
        verdict_dist = {"Synthetically-Generated" : sum(1 for r in filtered if r.verdict == "Synthetically-Generated"),
                        "Authentically-Written"   : sum(1 for r in filtered if r.verdict == "Authentically-Written"),
                        "Hybrid"                  : sum(1 for r in filtered if r.verdict == "Hybrid"),
                        "Uncertain"               : sum(1 for r in filtered if r.verdict == "Uncertain"),
                       }
        
        # Binary classification metrics (on decisive predictions only)
        if decisive:
            # Map predictions to binary (treating hybrid as AI detection)
            y_true_binary                        = [1 if r.ground_truth == "ai" else 0 for r in decisive]
            y_pred_binary                        = [1 if r.prediction in ["ai", "hybrid"] else 0 for r in decisive]
            
            # Calculate metrics
            precision, recall, f1, support_array = precision_recall_fscore_support(y_true_binary, 
                                                                                   y_pred_binary, 
                                                                                   average  = 'binary', 
                                                                                   pos_label = 1,
                                                                                   zero_division = 0,
                                                                                  )
            
            accuracy                             = sum(1 for i, r in enumerate(decisive) if y_true_binary[i] == y_pred_binary[i]) / len(decisive)
            
            # Confusion matrix
            cm                                   = confusion_matrix(y_true_binary, y_pred_binary)
            
            # Support counts
            support                              = {"human" : sum(1 for r in decisive if r.ground_truth == "human"),
                                                    "ai"    : sum(1 for r in decisive if r.ground_truth == "ai"),
                                                   }
        
        else:
            precision = recall = f1 = accuracy = 0.0
            cm                                 = [[0, 0], [0, 0]]
            support                            = {"human" : 0, "ai" : 0}
        
        # Probability-based metrics (on all samples with probabilities)
        y_true_prob = [1 if r.ground_truth == "ai" else 0 for r in filtered]
        y_scores    = [r.synthetic_prob for r in filtered]
        
        try:
            auroc = roc_auc_score(y_true_prob, y_scores)
        
        except:
            auroc = 0.0
        
        try:
            auprc = average_precision_score(y_true_prob, y_scores)
        
        except:
            auprc = 0.0
        
        # Expected Calibration Error (ECE)
        ece                    = self._calculate_ece(filtered)
        
        # Hybrid-specific metrics
        ai_samples             = [r for r in filtered if r.ground_truth == "ai"]
        hybrid_detection_rate  = sum(1 for r in ai_samples if r.prediction == "hybrid") / len(ai_samples) if ai_samples else 0.0
        
        # Abstention rate
        abstention_rate        = len(uncertain) / len(filtered) if filtered else 0.0
        
        return AggregatedMetrics(precision              = precision,
                                 recall                 = recall,
                                 f1                     = f1,
                                 accuracy               = accuracy,
                                 coverage               = coverage,
                                 accuracy_at_cov        = accuracy,
                                 f1_at_cov              = f1,
                                 auroc                  = auroc,
                                 auprc                  = auprc,
                                 ece                    = ece,
                                 hybrid_detection_rate  = hybrid_detection_rate,
                                 abstention_rate        = abstention_rate,
                                 confusion_matrix       = cm.tolist(),
                                 support                = support,
                                 verdict_distribution   = verdict_dist,
                                )
    

    def _calculate_ece(self, results: List[SingleEvalResult], n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error
        
        Arguments:
        ----------
            results { list } : List of evaluation results

            n_bins  { int }  : Number of confidence bins
            
        Returns:
        --------
            { float }        : ECE value
        """
        # Only calculate on decisive predictions
        decisive    = [r for r in results if r.prediction != "uncertain"]
        
        if not decisive:
            return 0.0
        
        confidences = np.array([r.confidence for r in decisive])
        predictions = np.array([1 if r.prediction in ["ai", "hybrid"] else 0 for r in decisive])
        labels      = np.array([1 if r.ground_truth == "ai" else 0 for r in decisive])
        
        ece         = 0.0
        
        for i in range(n_bins):
            bin_lower = i / n_bins
            bin_upper = (i + 1) / n_bins
            
            in_bin    = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if (np.sum(in_bin) > 0):
                bin_accuracy   = np.mean(predictions[in_bin] == labels[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_size       = np.sum(in_bin)
                
                ece           += (bin_size / len(decisive)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    

    def analyze_by_length(self) -> Dict[str, LengthBucketMetrics]:
        """
        Analyze performance across different text lengths
        
        Returns:
        --------
            { dict } : Dictionary mapping label to LengthBucketMetrics
        """
        # Define length buckets (word counts)
        length_buckets = [(0, 100, "Very Short (0-100)"),
                          (100, 200, "Short (100-200)"),
                          (200, 400, "Medium (200-400)"),
                          (400, 600, "Medium-Long (400-600)"),
                          (600, 1000, "Long (600-1000)"),
                          (1000, float("inf"), "Very Long (1000+)"),
                         ]
        
        bucket_metrics = dict()
        
        for min_words, max_words, label in length_buckets:
            # Filter results by length
            filtered              = [r for r in self.results if (min_words <= r.word_count < max_words)]
            
            if not filtered:
                continue
            
            # Separate abstained from decisive
            abstained             = [r for r in filtered if r.prediction in ["hybrid", "uncertain"]]
            decisive              = [r for r in filtered if r.prediction not in ["hybrid", "uncertain"]]
            
            if (len(decisive) < 5):
                continue

            # Calculate metrics for this bucket
            y_true                = [1 if (r.ground_truth == "ai") else 0 for r in decisive]
            y_pred                = [1 if (r.prediction == "ai") else 0 for r in decisive]
            
            if not y_true:
                continue
            
            # Precision, Recall, F1
            tp                    = sum(1 for i, _ in enumerate(y_true) if (y_true[i] == 1) and (y_pred[i] == 1))
            fp                    = sum(1 for i, _ in enumerate(y_true) if (y_true[i] == 0) and (y_pred[i] == 1))
            fn                    = sum(1 for i, _ in enumerate(y_true) if (y_true[i] == 1) and (y_pred[i] == 0))
            
            precision             = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall                = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1                    = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            valid                 = [r for r in decisive if r.is_correct is not None]
            accuracy              = sum(r.is_correct for r in valid) / len(valid) if valid else 0.0
            
            # Additional metrics
            mean_conf             = np.mean([r.confidence for r in decisive])
            fp_rate               = fp / len(decisive) if len(decisive) > 0 else 0.0
            fn_rate               = fn / len(decisive) if len(decisive) > 0 else 0.0
            avg_time              = np.mean([r.processing_time for r in decisive])
            
            bucket_metrics[label] = LengthBucketMetrics(min_words       = min_words,
                                                        max_words       = max_words,
                                                        label           = label,
                                                        sample_count    = len(decisive),
                                                        precision       = precision,
                                                        recall          = recall,
                                                        f1              = f1,
                                                        accuracy        = accuracy,
                                                        mean_confidence = mean_conf,
                                                        fp_rate         = fp_rate,
                                                        fn_rate         = fn_rate,
                                                        avg_proc_time   = avg_time,
                                                        abstention_rate = len(abstained) / (len(decisive) + len(abstained)) if (len(decisive) + len(abstained)) > 0 else 0.0,
                                                       )
        
        return bucket_metrics


    def print_length_analysis(self):
        """
        Print length-based performance analysis
        """
        print(f"\n{'=' * 80}")
        print("PERFORMANCE BY TEXT LENGTH")
        print("=" * 80)
        
        bucket_metrics = self.analyze_by_length()
        
        if not bucket_metrics:
            print("  No length analysis available")
            return
        
        print(f"\n{'Length Range':<25s} {'Samples':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} {'Accuracy':>10s} {'Abstain':>10s} {'Time(s)':>8s}")
        print("─" * 80)
        
        for label, metrics in bucket_metrics.items():
            print(f"{label:<25s} {metrics.sample_count:>8d} "
                  f"{metrics.f1:>8.3f} {metrics.precision:>10.3f} "
                  f"{metrics.recall:>8.3f} {metrics.accuracy:>10.3f} "
                  f"{metrics.abstention_rate:>10.2%} "
                  f"{metrics.avg_proc_time:>8.2f}"
                 )
        
        # Find best and worst performing buckets
        if (len(bucket_metrics) > 1):
            best  = max(bucket_metrics.items(), key = lambda x: x[1].f1)
            worst = min(bucket_metrics.items(), key = lambda x: x[1].f1)
            
            print(f"\n  Best Performance:  {best[0]} (F1: {best[1].f1:.3f})")
            print(f"  Worst Performance: {worst[0]} (F1: {worst[1].f1:.3f})")
        
        # Length-performance correlation
        self._analyze_length_correlation(bucket_metrics = bucket_metrics)


    def _analyze_length_correlation(self, bucket_metrics: Dict[str, LengthBucketMetrics]):
        """
        Analyze correlation between text length and performance
        
        Arguments:
        ----------
            bucket_metrics { dict } : Dictionary of length bucket metrics
        """
        if (len(bucket_metrics) < 3):
            return
        
        # Extract data for correlation
        lengths   = list()
        f1_scores = list()
        
        for metrics in bucket_metrics.values():
            # Skip buckets with too few samples
            if (metrics.sample_count < 5):
                continue

            # Skip degenerate buckets
            if ((metrics.f1 == 0.0) and (metrics.precision == 0.0) and (metrics.recall == 0.0)):
                continue

            # Representative length
            if np.isinf(metrics.max_words):
                avg_length = metrics.min_words

            else:
                avg_length = (metrics.min_words + metrics.max_words) / 2

            lengths.append(avg_length)
            f1_scores.append(metrics.f1)

        lengths   = np.asarray(lengths, dtype = float)
        f1_scores = np.asarray(f1_scores, dtype = float)

        # Statistical guards
        if (len(lengths) < 3):
            print("\n  Length-Performance Correlation:")
            print("    Skipped (insufficient valid buckets)")
            return

        if (not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(f1_scores))):
            print("\n  Length-Performance Correlation:")
            print("    Skipped (NaN / Inf detected)")
            return

        if (np.std(f1_scores) == 0.0):
            print("\n  Length-Performance Correlation:")
            print("    Skipped (zero variance in F1 scores)")
            return

        # Calculate Pearson correlation
        corr, p_value = stats.pearsonr(lengths, f1_scores)

        print(f"\n  Length-Performance Correlation:")
        print(f"    Pearson r = {corr:.3f} (p-value: {p_value:.4f})")

        if p_value < 0.05:
            if (corr > 0.3):
                print("    → Significant POSITIVE correlation — performance improves with length\n")

            elif (corr < -0.3):
                print("    → Significant NEGATIVE correlation — performance degrades with length\n")

            else:
                print("    → Weak but statistically significant correlation\n")
        
        else:
            print("    → No statistically significant correlation\n")


    def generate_report(self):
        """
        Generate comprehensive evaluation report
        """
        print("\n" + "=" * 70)
        print("EVALUATION REPORT - 4-CLASS SYSTEM")
        print("=" * 70)
        
        # Overall metrics
        overall = self.calculate_metrics()
        
        if overall:
            print("\nOverall Performance (Decisive Predictions):")
            print(f"  Coverage: {overall.coverage:.1%} (decisive predictions)")
            print(f"  Accuracy: {overall.accuracy:.1%}")
            print(f"  Precision (AI): {overall.precision:.1%}")
            print(f"  Recall (AI): {overall.recall:.1%}")
            print(f"  F1 Score: {overall.f1:.1%}")
            print(f"  AUROC: {overall.auroc:.3f}")
            print(f"  AUPRC: {overall.auprc:.3f}")
            print(f"  ECE (Calibration): {overall.ece:.3f}")
            
            print(f"\n4-Class Specific Metrics:")
            print(f"  Abstention Rate: {overall.abstention_rate:.1%}")
            print(f"  Hybrid Detection Rate: {overall.hybrid_detection_rate:.1%}")
            
            print("\n  Verdict Distribution:")
            for verdict, count in overall.verdict_distribution.items():
                pct = count / len(self.results) * 100
                print(f"    {verdict:30s}: {count:4d} ({pct:5.1f}%)")
        
        # Per-domain performance
        print("\n" + "-" * 70)
        print("Per-Domain Performance:")
        print("-" * 70)
        print(f"{'Domain':<20s} {'F1':>8s} {'Coverage':>10s} {'Abstain':>10s} {'Hybrid%':>10s}")
        print("-" * 70)
        
        domain_scores = list()
        
        for domain in sorted(set(r.domain for r in self.results)):
            metrics = self.calculate_metrics(domain = domain)
            
            if metrics and (metrics.support['ai'] + metrics.support['human']) >= 5:
                domain_scores.append((domain, metrics.f1, metrics.coverage))
                print(f"{domain:<20s} {metrics.f1:>8.1%} {metrics.coverage:>10.1%} {metrics.abstention_rate:>10.1%} {metrics.hybrid_detection_rate:>10.1%}")
        
        # Per-subset performance
        print("\n" + "-" * 70)
        print("Per-Subset Performance:")
        print("-" * 70)
        
        for subset in sorted(set(r.subset for r in self.results)):
            metrics = self.calculate_metrics(subset = subset)
            
            if metrics:
                print(f"\n  {subset.upper()}:")
                print(f"    Samples: {metrics.support['human'] + metrics.support['ai']}")
                print(f"    F1 Score: {metrics.f1:.1%}")
                print(f"    Coverage: {metrics.coverage:.1%}")
                print(f"    Abstention: {metrics.abstention_rate:.1%}")
                print(f"    Hybrid Detection: {metrics.hybrid_detection_rate:.1%}")
        
        # Length analysis
        self.print_length_analysis()
        
        print("\n" + "=" * 70)
    

    def save_results(self):
        """
        Save evaluation results
        """
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate overall metrics
        overall          = self.calculate_metrics()
        
        # Length analysis
        length_metrics   = self.analyze_by_length()
        length_dict      = {k: asdict(v) for k, v in length_metrics.items()}
        
        # Abstention stats
        abstention_stats = {"total_uncertain"  : sum(1 for r in self.results if r.prediction == "uncertain"),
                            "total_hybrid"     : sum(1 for r in self.results if r.prediction == "hybrid"),
                            "abstention_rate"  : sum(1 for r in self.results if r.prediction == "uncertain") / len(self.results) if self.results else 0,
                            "hybrid_rate"      : sum(1 for r in self.results if r.prediction == "hybrid") / len(self.results) if self.results else 0,
                            "avg_uncertainty"  : np.mean([r.uncertainty for r in self.results if r.prediction == "uncertain"]) if any(r.prediction == "uncertain" for r in self.results) else 0,
                           }
        
        # Save detailed results as JSON
        results_dict     = [asdict(r) for r in self.results]
        json_path        = self.output_dir / f"evaluation_results_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(obj    = {'metadata'        : self.metadata,
                                'overall_metrics' : asdict(overall) if overall else {},
                                'length_metrics'  : length_dict,
                                'abstention'      : abstention_stats,
                                'timestamp'       : timestamp,
                                'results'         : results_dict,
                               }, 
                      fp     = f, 
                      indent = 4,
                     )

        print(f"\n✓ JSON results saved: {json_path}")
        
        # Save as CSV
        df       = pd.DataFrame(data = results_dict)
        csv_path = self.output_dir / f"evaluation_results_{timestamp}.csv"
        
        df.to_csv(csv_path, index = False)
        print(f"✓ CSV results saved: {csv_path}")
    

    def plot_visualizations(self):
        """
        Generate comprehensive evaluation visualizations
        """
        fig, axes = plt.subplots(nrows   = 2,
                                 ncols   = 2,
                                 figsize = (18, 14),
                                )

        plt.suptitle('TEXT-AUTH Evaluation Results (4-Class System)',
                     fontsize   = 18,
                     fontweight = 'bold',
                     y          = 0.98,
                    )

        # Confusion Matrix (decisive predictions only)
        ax1      = axes[0, 0]
        overall  = self.calculate_metrics()
        
        if overall:
            cm   = np.array(overall.confusion_matrix)

            sns.heatmap(cm,
                        annot       = True,
                        fmt         = 'd',
                        cmap        = 'Blues',
                        ax          = ax1,
                        xticklabels = ['Human', 'AI/Hybrid'],
                        yticklabels = ['Human', 'AI'],
                       )

            ax1.set_title('Confusion Matrix\n(Decisive Predictions Only)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')


        # F1 Score by Domain
        ax2           = axes[0, 1]
        domain_scores = list()

        for domain in sorted(set(r.domain for r in self.results)):
            metrics = self.calculate_metrics(domain = domain)

            if metrics and (metrics.support['ai'] + metrics.support['human']) >= 10:
                domain_scores.append((domain, metrics.f1))

        domain_scores.sort(key = lambda x: x[1])

        if domain_scores:
            domain_labels, domain_f1 = zip(*domain_scores)

            ax2.barh(domain_labels, domain_f1, color = 'steelblue')

            if overall:
                ax2.axvline(x         = overall.f1,
                            color     = 'red',
                            linestyle = '--',
                            linewidth = 1.5,
                            label     = f'Overall ({overall.f1:.1%})',
                           )

            ax2.set_xlim([0, 1])
            ax2.set_xlabel('F1 Score')
            ax2.set_title('F1 Score by Domain')
            ax2.grid(axis = 'x', alpha = 0.3)
            ax2.legend()


        # Verdict Distribution (Pie Chart)
        ax3 = axes[1, 0]
        
        if overall:
            verdict_counts = overall.verdict_distribution
            labels         = list(verdict_counts.keys())
            sizes          = list(verdict_counts.values())
            colors         = ['#fee2e2', '#d1fae5', '#e9d5ff', '#fef3c7']
            explode        = (0.05, 0.05, 0.05, 0.05)
            
            ax3.pie(sizes, 
                   labels   = labels, 
                   autopct  = '%1.1f%%', 
                   colors   = colors, 
                   explode  = explode,
                   startangle = 90,
                   textprops = {'fontsize': 9})
            ax3.set_title('4-Class Verdict Distribution')


        # Robustness by Subset
        ax4           = axes[1, 1]
        subset_scores = list()

        for subset in sorted(set(r.subset for r in self.results)):
            metrics = self.calculate_metrics(subset = subset)

            if metrics:
                subset_scores.append((subset, metrics.f1, metrics.coverage, metrics.abstention_rate))

        if subset_scores:
            subset_labels = [s[0] for s in subset_scores]
            subset_f1     = [s[1] for s in subset_scores]
            subset_cov    = [s[2] for s in subset_scores]
            subset_abs    = [s[3] for s in subset_scores]

            x             = np.arange(len(subset_labels))
            width         = 0.25

            ax4.bar(x - width, subset_f1, width, label = 'F1 Score', color = 'steelblue')
            ax4.bar(x, subset_cov, width, label = 'Coverage', color = 'lightcoral')
            ax4.bar(x + width, subset_abs, width, label = 'Abstention', color = 'gold')

            ax4.set_ylabel('Score / Rate')
            ax4.set_title('Performance & Behavior by Subset')
            ax4.set_xticks(x)
            ax4.set_xticklabels(subset_labels, rotation = 45, ha = 'right')
            ax4.legend()
            ax4.grid(axis = 'y', alpha = 0.3)
            ax4.set_ylim([0, 1])


        plt.tight_layout(rect = [0, 0, 1, 0.96])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"evaluation_plots_{timestamp}.png"

        plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
        plt.close()

        print(f"✓ Main plots saved: {plot_path}")
        
        # Generate length analysis plots
        self.plot_length_visualizations()


    def plot_length_visualizations(self):
        """
        Generate length-based performance visualizations
        """
        bucket_metrics = self.analyze_by_length()
        
        if not bucket_metrics or len(bucket_metrics) < 2:
            return
        
        fig, axes      = plt.subplots(nrows   = 2,
                                      ncols   = 2,
                                      figsize = (16, 12),
                                     )

        plt.suptitle('Performance Analysis by Text Length',
                     fontsize   = 16,
                     fontweight = 'bold',
                     y          = 0.98,
                    )

        labels         = list(bucket_metrics.keys())
        metrics_list   = list(bucket_metrics.values())

        # F1, Precision, Recall by Length
        ax1            = axes[0, 0]
        
        f1_vals        = [m.f1 for m in metrics_list]
        precision_vals = [m.precision for m in metrics_list]
        recall_vals    = [m.recall for m in metrics_list]
        
        x              = np.arange(len(labels))
        width          = 0.25
        
        ax1.bar(x - width, f1_vals, width, label = 'F1', color = 'steelblue')
        ax1.bar(x, precision_vals, width, label = 'Precision', color = 'lightcoral')
        ax1.bar(x + width, recall_vals, width, label = 'Recall', color = 'lightgreen')
        
        ax1.set_ylabel('Score')
        ax1.set_title('Classification Metrics by Length')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation = 45, ha = 'right', fontsize = 9)
        ax1.legend()
        ax1.grid(axis = 'y', alpha = 0.3)
        ax1.set_ylim([0, 1])

        # Sample Distribution
        ax2           = axes[0, 1]
        
        sample_counts = [m.sample_count for m in metrics_list]
        
        ax2.bar(labels, sample_counts, color = 'mediumpurple')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Sample Distribution by Length')
        ax2.set_xticklabels(labels, rotation = 45, ha = 'right', fontsize = 9)
        ax2.grid(axis = 'y', alpha = 0.3)

        # Processing Time by Length
        ax3        = axes[1, 0]
        
        proc_times = [m.avg_proc_time for m in metrics_list]
        
        ax3.plot(labels, proc_times, marker = 'o', linewidth = 2, markersize = 8, color = 'darkorange')
        ax3.set_ylabel('Processing Time (seconds)')
        ax3.set_title('Average Processing Time by Length')
        ax3.set_xticklabels(labels, rotation = 45, ha = 'right', fontsize = 9)
        ax3.grid(alpha = 0.3)

        # Abstention Rate by Length
        ax4              = axes[1, 1]
        
        abstention_rates = [m.abstention_rate * 100 for m in metrics_list]
        
        ax4.bar(labels, abstention_rates, color = 'gold')
        ax4.set_ylabel('Abstention Rate (%)')
        ax4.set_title('Abstention Rate by Length')
        ax4.set_xticklabels(labels, rotation = 45, ha = 'right', fontsize = 9)
        ax4.grid(axis = 'y', alpha = 0.3)

        plt.tight_layout(rect = [0, 0, 1, 0.96])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"length_analysis_{timestamp}.png"

        plt.savefig(plot_path, dpi = 300, bbox_inches = 'tight')
        plt.close()

        print(f"✓ Length analysis plots saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description = 'Run TEXT-AUTH evaluation (4-class system)')
    parser.add_argument('--dataset', type = str, default = 'evaluation', help = 'Path to evaluation directory')
    parser.add_argument('--output', type = str, default = 'evaluation/results', help = 'Output directory for results')
    parser.add_argument('--quick-test', action = 'store_true', help = 'Run quick test on 10 samples per domain')
    parser.add_argument('--samples', type = int, default = None, help = 'Maximum samples per domain')
    parser.add_argument('--domains', type = str, nargs = '+', default = None, help = 'Specific domains to evaluate')
    parser.add_argument('--subset', type = str, choices = ['clean', 'paraphrased', 'cross_model'], help = 'Evaluate only specific subset')
    
    args        = parser.parse_args()
    
    # Initialize evaluator
    evaluator   = TextAuthEvaluator(dataset_path = args.dataset,
                                    output_dir   = args.output,
                                   )
    
    # Load dataset
    max_samples = 10 if args.quick_test else args.samples
    samples     = evaluator.load_dataset(domains                = args.domains,
                                         max_samples_per_domain = max_samples,
                                         subset_filter          = args.subset,
                                        )
    
    if not samples:
        print("No samples loaded. Check dataset path and run data collection scripts.")
        return 1
    
    # Run evaluation
    evaluator.run_evaluation(samples)
    
    # Generate report 
    evaluator.generate_report()
    
    # Save results
    evaluator.save_results()
    
    # Generate plots
    evaluator.plot_visualizations()
    
    print("\n✓ Evaluation complete!\n")
    return 0



# Execution
if __name__ == "__main__":
    sys.exit(main())