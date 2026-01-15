# Dependencies
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from typing import Tuple
from datetime import datetime
from dataclasses import asdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.enums import Domain
from config.schemas import DetectionResult
from services.orchestrator import DetectionOrchestrator
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class EvaluationSample:
    """
    Single evaluation sample
    """
    text_id       : str
    domain        : str
    ground_truth  : str  # "human" or "ai"
    text          : str
    file_path     : str
    subset        : str  # "clean", "paraphrased", "cross_model"


@dataclass
class SingleEvalResult:
    """
    Result for a single sample
    """
    text_id         : str
    domain          : str
    ground_truth    : str
    prediction      : str  # "human" or "ai"
    verdict         : str  # Raw verdict from system
    synthetic_prob  : float
    authentic_prob  : float
    confidence      : float
    processing_time : float
    is_correct      : bool
    subset          : str


@dataclass
class AggregatedMetrics:
    """
    Aggregated performance metrics
    """
    precision        : float
    recall           : float
    f1               : float
    accuracy         : float
    auroc            : float
    auprc            : float
    ece              : float           # Expected Calibration Error
    confusion_matrix : List[List[int]]
    support          : Dict[str, int]


class TextAuthEvaluator:
    """
    Comprehensive evaluation framework for TEXT-AUTH
    """
    def __init__(self, dataset_path: str, output_dir: str = "evaluation/results"):
        """
        Initialize evaluator
        
        Arguments:
        ----------
            dataset_path { Path } : Path to TEXT-AUTH evaluation/ dataset

            output_dir   { Path } : Directory to save results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir   = Path(output_dir)

        self.output_dir.mkdir(exist_ok = True, 
                              parents  = True,
                             )
        
        # Initialize orchestrator
        print("Initializing TEXT-AUTH Detection Orchestrator...")
        self.orchestrator = DetectionOrchestrator.create_with_executor(max_workers               = 4,
                                                                       enable_language_detection = False,
                                                                       parallel_execution        = True,
                                                                       skip_expensive_metrics    = False,
                                                                      )
        
        if not self.orchestrator.initialize():
            raise RuntimeError("Failed to initialize detection orchestrator")
        
        print("Orchestrator initialized successfully")
        
        # Storage for results
        self.results  : List[SingleEvalResult] = list()
        self.metadata : Dict[str, Any]         = dict()
        
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

            print(f"Loaded metadata: {self.metadata.get('dataset_name', 'Unknown')}")

        else:
            print("⚠️ No metadata.json found")
    
    def load_dataset(self, domains: List[str] = None, max_samples_per_domain: int = None) -> List[EvaluationSample]:
        """
        Load evaluation dataset
        
        Arguments:
        ----------
            domains               { list } : List of domains to evaluate (None = all)

            max_samples_per_domain { int } : Limit samples per domain
            
        Returns:
        --------
                     { list }              : List of EvaluationSample objects
        """
        samples = list()
        
        # Load clean samples (human + ai)
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
                        print(f"⚠️ Error loading {file_path}: {e}")
        
        # Load challenge sets (adversarial)
        adversarial_path = self.dataset_path / "adversarial"
        
        if adversarial_path.exists():
            for challenge_dir in adversarial_path.iterdir():
                if not challenge_dir.is_dir():
                    continue
                
                subset_name = challenge_dir.name  # "paraphrased" or "cross_model"
                
                for file_path in challenge_dir.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Extract domain from filename if possible
                        domain = "general"
                        
                        for d in ["academic", "technical", "creative", "business", "legal"]:
                            if d in file_path.stem.lower():
                                domain = d
                                break
                        
                        samples.append(EvaluationSample(text_id      = file_path.stem,
                                                        domain       = domain,
                                                        ground_truth = "ai",  # Challenge sets are AI-generated
                                                        text         = text,
                                                        file_path    = str(file_path),
                                                        subset       = subset_name,
                                                       )
                                      )

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        print(f"\nLoaded {len(samples)} samples:")
        print(f"  - Clean samples: {sum(1 for s in samples if s.subset == 'clean')}")
        print(f"  - Paraphrased: {sum(1 for s in samples if s.subset == 'paraphrased')}")
        print(f"  - Cross-model: {sum(1 for s in samples if s.subset == 'cross_model')}")
        
        return samples
    

    def evaluate_single(self, sample: EvaluationSample) -> SingleEvalResult:
        """
        Evaluate single sample
        """
        start_time = time.time()
        
        try:
            # Parse domain
            domain_enum                = self._parse_domain(sample.domain)
            
            # Run detection
            detection: DetectionResult = self.orchestrator.analyze(text   = sample.text,
                                                                   domain = domain_enum,
                                                                  )
            
            processing_time            = time.time() - start_time
            
            # Extract results
            verdict                    = detection.ensemble_result.final_verdict
            synthetic_prob             = detection.ensemble_result.synthetic_probability
            authentic_prob             = detection.ensemble_result.authentic_probability
            confidence                 = detection.ensemble_result.overall_confidence
            
            # Map verdict to prediction
            if (verdict == "Synthetically-Generated"):
                prediction = "ai"

            elif (verdict == "Authentically-Written"):
                prediction = "human"

            else: 
                # Use probability to decide: "Hybrid" or "Uncertain"
                prediction = "ai" if (synthetic_prob > authentic_prob) else "human"
            
            is_correct = (prediction == sample.ground_truth)
            
            return SingleEvalResult(text_id         = sample.text_id,
                                    domain          = sample.domain,
                                    ground_truth    = sample.ground_truth,
                                    prediction      = prediction,
                                    verdict         = verdict,
                                    synthetic_prob  = synthetic_prob,
                                    authentic_prob  = authentic_prob,
                                    confidence      = confidence,
                                    processing_time = processing_time,
                                    is_correct      = is_correct,
                                    subset          = sample.subset,
                                   )
            
        except Exception as e:
            print(f"Error evaluating {sample.text_id}: {e}")

            return SingleEvalResult(text_id         = sample.text_id,
                                    domain          = sample.domain,
                                    ground_truth    = sample.ground_truth,
                                    prediction      = "error",
                                    verdict         = "Error",
                                    synthetic_prob  = 0.5,
                                    authentic_prob  = 0.5,
                                    confidence      = 0.0,
                                    processing_time = 0.0,
                                    is_correct      = False,
                                    subset          = sample.subset,
                                   )
    

    def run_evaluation(self, samples: List[EvaluationSample], show_progress: bool = True):
        """
        Run evaluation on all samples
        """
        print(f"\n🔬 Evaluating {len(samples)} samples...")
        
        iterator = tqdm(samples, desc = "Evaluating") if show_progress else samples
        
        for sample in iterator:
            result = self.evaluate_single(sample)
            self.results.append(result)
        
        print(f"\nEvaluation complete: {len(self.results)} samples processed")
    

    def calculate_metrics(self, subset: str = None, domain: str = None) -> AggregatedMetrics:
        """
        Calculate aggregated metrics
        """
        # Filter results
        filtered = self.results

        if subset:
            filtered = [r for r in filtered if (r.subset == subset)]

        if domain:
            filtered = [r for r in filtered if (r.domain == domain)]
        
        if not filtered:
            return None
        
        # Extract arrays
        y_true                         = [1 if (r.ground_truth == "ai") else 0 for r in filtered]
        y_pred                         = [1 if (r.prediction == "ai") else 0 for r in filtered]
        y_prob                         = [r.synthetic_prob for r in filtered]
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true        = y_true, 
                                                                         y_pred        = y_pred, 
                                                                         average       = 'binary', 
                                                                         zero_division = 0,
                                                                        )
        
        accuracy                       = sum([r.is_correct for r in filtered]) / len(filtered)
        
        # AUROC
        try:
            auroc = roc_auc_score(y_true  = y_true, 
                                  y_score = y_prob,
                                 )


        except:
            auroc = 0.0
        
        # AUPRC
        try:
            pr_precision, pr_recall, _ = precision_recall_curve(y_true  = y_true, 
                                                                y_score = y_prob,
                                                               )

            auprc                      = auc(x = pr_recall, 
                                             y = pr_precision,
                                            )

        except:
            auprc = 0.0
        
        # ECE (Expected Calibration Error)
        ece          = self._calculate_ece(y_true = y_true, 
                                           y_prob = y_prob,
                                          )
        
        # Confusion matrix
        cm           = confusion_matrix(y_true = y_true, 
                                        y_pred = y_pred,
                                       ).tolist()
        
        # Support counts
        support_dict = {"human" : sum(1 for r in filtered if r.ground_truth == "human"),
                        "ai"    : sum(1 for r in filtered if r.ground_truth == "ai"),
                       }
        
        return AggregatedMetrics(precision        = precision,
                                 recall           = recall,
                                 f1               = f1,
                                 accuracy         = accuracy,
                                 auroc            = auroc,
                                 auprc            = auprc,
                                 ece              = ece,
                                 confusion_matrix = cm,
                                 support          = support_dict,
                                )


    def _calculate_ece(self, y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error
        """
        bins        = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        
        ece         = 0.0

        for i in range(n_bins):
            mask = (bin_indices == i)

            if (mask.sum() > 0):
                bin_acc    = np.array(y_true)[mask].mean()
                bin_conf   = np.array(y_prob)[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece       += abs(bin_acc - bin_conf) * bin_weight
        
        return ece

    
    def _parse_domain(self, domain_str: str) -> Domain:
        """
        Parse domain string to enum
        """
        domain_map = {"general"       : Domain.GENERAL,
                      "academic"      : Domain.ACADEMIC,
                      "creative"      : Domain.CREATIVE,
                      "ai_ml"         : Domain.AI_ML,
                      "software_dev"  : Domain.SOFTWARE_DEV,
                      "technical_doc" : Domain.TECHNICAL_DOC,
                      "engineering"   : Domain.ENGINEERING,
                      "science"       : Domain.SCIENCE,
                      "business"      : Domain.BUSINESS,
                      "legal"         : Domain.LEGAL,
                      "medical"       : Domain.MEDICAL,
                      "journalism"    : Domain.JOURNALISM,
                      "marketing"     : Domain.MARKETING,
                      "social_media"  : Domain.SOCIAL_MEDIA,
                      "blog_personal" : Domain.BLOG_PERSONAL,
                      "tutorial"      : Domain.TUTORIAL,
                     }

        return domain_map.get(domain_str.lower(), Domain.GENERAL)
    

    def generate_report(self):
        """
        Generate comprehensive evaluation report
        """
        print("\n" + "="*80)
        print("TEXT-AUTH EVALUATION REPORT")
        print("="*80)
        
        # Overall metrics
        print(f"\n📊 OVERALL PERFORMANCE")
        overall = self.calculate_metrics()

        self._print_metrics(overall)
        
        # Per-subset metrics
        print(f"\n📈 PERFORMANCE BY SUBSET")
        for subset in ["clean", "paraphrased", "cross_model"]:
            subset_results = [r for r in self.results if r.subset == subset]
            
            if subset_results:
                print(f"\n{subset.upper()} ({len(subset_results)} samples):")
                metrics = self.calculate_metrics(subset=subset)
                
                self._print_metrics(metrics, indent=2)
        
        # Per-domain metrics
        print(f"\n📈 PERFORMANCE BY DOMAIN")
        
        domains = set(r.domain for r in self.results)
        
        for domain in sorted(domains):
            domain_results = [r for r in self.results if r.domain == domain]

            # Only show if sufficient samples
            if (len(domain_results) >= 10):  
                print(f"\n{domain.upper()} ({len(domain_results)} samples):")
                
                metrics = self.calculate_metrics(domain = domain)
                
                self._print_metrics(metrics, indent=2)
        
        # Failure analysis
        print(f"\n❌ FAILURE ANALYSIS")
        self._print_failure_analysis()
        
        # Performance metrics
        avg_time = np.mean([r.processing_time for r in self.results])
        p95_time = np.percentile([r.processing_time for r in self.results], 95)
        
        print(f"\n⚡ PERFORMANCE")
        print(f"  Mean processing time:  {avg_time:.2f}s")
        print(f"  P95 processing time:   {p95_time:.2f}s")
        print(f"  Total samples:         {len(self.results)}")
        
        print("\n" + "="*80)
    

    def _print_metrics(self, metrics: AggregatedMetrics, indent: int = 0):
        """
        Print metrics with indentation
        """
        prefix = " " * indent
        print(f"{prefix}├─ Precision:  {metrics.precision:.3f}")
        print(f"{prefix}├─ Recall:     {metrics.recall:.3f}")
        print(f"{prefix}├─ F1:         {metrics.f1:.3f}")
        print(f"{prefix}├─ Accuracy:   {metrics.accuracy:.3f}")
        print(f"{prefix}├─ AUROC:      {metrics.auroc:.3f}")
        print(f"{prefix}├─ AUPRC:      {metrics.auprc:.3f}")
        print(f"{prefix}└─ ECE:        {metrics.ece:.3f}")
    

    def _print_failure_analysis(self):
        """
        Print failure analysis
        """
        false_negatives = [r for r in self.results if (r.ground_truth == "ai") and (r.prediction == "human")]
        false_positives = [r for r in self.results if (r.ground_truth == "human") and (r.prediction == "ai")]
        
        print(f"  False Negatives (AI→Human): {len(false_negatives)} ({len(false_negatives)/len(self.results)*100:.1f}%)")
        print(f"  False Positives (Human→AI): {len(false_positives)} ({len(false_positives)/len(self.results)*100:.1f}%)")
        
        # Show a few examples
        if false_negatives:
            print(f"\n  Example False Negatives:")
            for r in false_negatives[:3]:
                print(f"    - {r.text_id} ({r.domain}): conf={r.confidence:.2f}, syn_prob={r.synthetic_prob:.2f}")
        
        if false_positives:
            print(f"\n  Example False Positives:")
            for r in false_positives[:3]:
                print(f"    - {r.text_id} ({r.domain}): conf={r.confidence:.2f}, syn_prob={r.synthetic_prob:.2f}")
    

    def save_results(self):
        """
        Save evaluation results
        """
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_dict = [asdict(r) for r in self.results]
        json_path    = self.output_dir / f"evaluation_results_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(obj  = {'metadata'        : self.metadata,
                              'overall_metrics' : asdict(self.calculate_metrics()),
                              'results'         : results_dict,
                             }, 
                    fp     = f, 
                    indent = 2,
                   )

        print(f"Results saved to: {json_path}")
        
        # Save as CSV for easy analysis
        df       = pd.DataFrame(data = results_dict)
        csv_path = self.output_dir / f"evaluation_results_{timestamp}.csv"

        df.to_csv(csv_path, index = False)

        print(f"CSV saved to: {csv_path}")
    

    def plot_visualizations(self):
        """
        Generate evaluation visualizations
        """
        fig, axes = plt.subplots(nrows   = 2, 
                                 ncols   = 2, 
                                 figsize = (15, 12),
                                )
        
        # Confusion Matrix
        ax1       = axes[0, 0]
        overall   = self.calculate_metrics()
        cm        = overall.confusion_matrix

        sns.heatmap(cm, 
                    annot = True, 
                    fmt   = 'd', 
                    cmap  = 'Blues', 
                    ax    = ax1,
                   )

        ax1.set_title('Overall Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticklabels(['Human', 'AI'])
        ax1.set_yticklabels(['Human', 'AI'])
        
        # Performance by Domain
        ax2       = axes[0, 1]
        domains   = sorted(set(r.domain for r in self.results))
        f1_scores = list()

        for domain in domains:
            metrics = self.calculate_metrics(domain=domain)
            if metrics:
                f1_scores.append(metrics.f1)
            
            else:
                f1_scores.append(0)

        ax2.barh(domains, f1_scores, color = 'steelblue')
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score by Domain')
        ax2.set_xlim([0, 1])
        
        # Confidence Distribution
        ax3       = axes[1, 0]
        correct   = [r.confidence for r in self.results if r.is_correct]
        incorrect = [r.confidence for r in self.results if not r.is_correct]

        ax3.hist(correct, 
                 bins  = 20, 
                 alpha = 0.6, 
                 label = 'Correct', 
                 color = 'green',
                )

        ax3.hist(incorrect, 
                 bins  = 20, 
                 alpha = 0.6, 
                 label = 'Incorrect', 
                 color = 'red',
                )

        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Confidence Distribution')
        ax3.legend()
        
        # Performance by Subset
        ax4          = axes[1, 1]
        subsets      = ['clean', 'paraphrased', 'cross_model']
        f1_by_subset = list()

        for subset in subsets:
            metrics = self.calculate_metrics(subset=subset)
            if metrics:
                f1_by_subset.append(metrics.f1)
            
            else:
                f1_by_subset.append(0)
        
        ax4.bar(subsets, 
                f1_by_subset, 
                color = ['green', 'orange', 'red'],
               )

        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score by Challenge Set')
        ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"evaluation_plots_{timestamp}.png"

        plt.savefig(fname       = plot_path, 
                    dpi         = 300, 
                    bbox_inches = 'tight',
                   )

        print(f"Visualizations saved to: {plot_path}")

        plt.close()


# Main Entry Point
def main():
    parser = argparse.ArgumentParser(description = 'Run TEXT-AUTH evaluation')
    
    parser.add_argument('--dataset', type = str, default = 'evaluation', help = 'Path to evaluation dataset')
    parser.add_argument('--output', type  = str, default = 'evaluation/results', help = 'Output directory for results')
    parser.add_argument('--quick-test', action = 'store_true', help = 'Run quick test on limited samples')
    parser.add_argument('--samples', type = int, default = None, help = 'Maximum samples per domain')
    parser.add_argument('--domains', type = str, nargs = '+', default = None, help = 'Specific domains to evaluate')
    
    args      = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TextAuthEvaluator(dataset_path = args.dataset,
                                  output_dir   = args.output,
                                 )
    
    # Load dataset
    max_samples = 10 if args.quick_test else args.samples
    samples     = evaluator.load_dataset(domains                = args.domains,
                                         max_samples_per_domain = max_samples,
                                        )
    
    if not samples:
        print("No samples loaded. Check dataset path.")
        return
    
    # Run evaluation
    evaluator.run_evaluation(samples)
    
    # Generate report
    evaluator.generate_report()
    
    # Save results
    evaluator.save_results()
    
    # Generate plots
    evaluator.plot_visualizations()
    
    print("\nEvaluation complete!")


# Execute Evaluation
if __name__ == "__main__":
    main()