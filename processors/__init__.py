# DEPENDENCIES
from .text_processor import *
from .language_detector import *
from .domain_classifier import *
from .document_extractor import *


# Export everything
__all__ = ["Script", 
           "Language",
           "is_english",
           "extract_text",
           "quick_detect", 
           "TextProcessor",
           "ProcessedText", 
           "quick_process",
           "extract_words",
           "LanguageDetector",
           "DomainClassifier", 
           "DomainPrediction", 
           "extract_sentences",
           "DocumentExtractor",
           "ExtractedDocument", 
           "extract_from_upload",
           "LanguageDetectionResult", 
          ]     