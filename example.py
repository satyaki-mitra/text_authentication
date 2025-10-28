# Complete detection + reporting pipeline

from detector.orchestrator import DetectionOrchestrator
from detector.attribution import ModelAttributor
from reporter.report_generator import ReportGenerator

# 1. Initialize components
orchestrator = DetectionOrchestrator()
orchestrator.initialize()

attributor = ModelAttributor()
attributor.initialize()

reporter = ReportGenerator()

# 2. Analyze text
text = """Perplexity measures how well a language model predicts a sample; lower perplexity indicates better predictive accuracy. In AI detection, models often exhibit unnaturally low perplexity because their outputs are statistically optimized rather than organically generated. Human writing tends to have higher variability and “burstiness”—irregular patterns of word choice and sentence structure. By combining perplexity with burstiness analysis and fine-tuned classifiers like RoBERTa, detectors can identify AI-generated text with greater confidence. Ensemble methods further improve reliability by aggregating multiple signals. This multi-layered approach reduces false positives and adapts to evolving AI models. Understanding these metrics helps users interpret detection scores meaningfully."""

detection_result = orchestrator.analyze(text)

# 3. Attribute model
attribution_result = attributor.attribute(
    text=text,
    processed_text=detection_result.processed_text,
    metric_results=detection_result.metric_results,
)

# 4. Generate reports
report_files = reporter.generate_complete_report(
    detection_result=detection_result,
    attribution_result=attribution_result,
    formats=["json", "pdf", "txt"],
    filename_prefix="my_analysis",
)

print("Generated reports:")
for format_type, filepath in report_files.items():
    print(f"  {format_type.upper()}: {filepath}")

# Output:
# Generated reports:
#   JSON: reports/output/my_analysis_20250101_143022.json
#   HTML: reports/output/my_analysis_20250101_143022.html
#   PDF: reports/output/my_analysis_20250101_143022.pdf
#   TXT: reports/output/my_analysis_20250101_143022.txt