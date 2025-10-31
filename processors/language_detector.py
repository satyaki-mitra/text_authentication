# DEPENDENCIES
import re
import torch
import string
from enum import Enum
from typing import Dict
from typing import List
from typing import Tuple
from loguru import logger
from typing import Optional
from dataclasses import dataclass


# Try to import optional libraries
try:
    import langdetect
    from langdetect import detect, detect_langs, DetectorFactory
    # Seed for reproducibility
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True

except ImportError:
    logger.warning("langdetect not available. Install: pip install langdetect")
    LANGDETECT_AVAILABLE = False

try:
    from models.model_manager import get_model_manager
    MODEL_MANAGER_AVAILABLE = True
    
except ImportError:
    logger.warning("model_manager not available, using fallback methods")
    MODEL_MANAGER_AVAILABLE = False


class Language(Enum):
    """
    ISO 639-1 language codes for supported languages
    """
    ENGLISH    = "en"
    SPANISH    = "es"
    FRENCH     = "fr"
    GERMAN     = "de"
    ITALIAN    = "it"
    PORTUGUESE = "pt"
    RUSSIAN    = "ru"
    CHINESE    = "zh"
    JAPANESE   = "ja"
    KOREAN     = "ko"
    ARABIC     = "ar"
    HINDI      = "hi"
    DUTCH      = "nl"
    POLISH     = "pl"
    TURKISH    = "tr"
    SWEDISH    = "sv"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    THAI       = "th"
    GREEK      = "el"
    HEBREW     = "he"
    CZECH      = "cs"
    ROMANIAN   = "ro"
    DANISH     = "da"
    FINNISH    = "fi"
    NORWEGIAN  = "no"
    UNKNOWN    = "unknown"


class Script(Enum):
    """
    Writing scripts
    """
    LATIN      = "latin"
    CYRILLIC   = "cyrillic"
    ARABIC     = "arabic"
    CHINESE    = "chinese"
    JAPANESE   = "japanese"
    KOREAN     = "korean"
    DEVANAGARI = "devanagari"
    GREEK      = "greek"
    HEBREW     = "hebrew"
    THAI       = "thai"
    MIXED      = "mixed"
    UNKNOWN    = "unknown"


@dataclass
class LanguageDetectionResult:
    """
    Result of language detection
    """
    primary_language   : Language
    confidence         : float
    all_languages      : Dict[str, float]  # language_code -> confidence
    script             : Script
    is_multilingual    : bool
    detection_method   : str
    char_count         : int
    word_count         : int
    warnings           : List[str]
    

    def to_dict(self) -> Dict:
        """
        Convert to dictionary
        """
        return {"primary_language"  : self.primary_language.value,
                "confidence"        : round(self.confidence, 4),
                "all_languages"     : {k: round(v, 4) for k, v in self.all_languages.items()},
                "script"            : self.script.value,
                "is_multilingual"   : self.is_multilingual,
                "detection_method"  : self.detection_method,
                "char_count"        : self.char_count,
                "word_count"        : self.word_count,
                "warnings"          : self.warnings,
               }


class LanguageDetector:
    """
    Detects the language of input text using multiple strategies with fallbacks.

    Features:
    - Primary    : XLM-RoBERTa model (supports 100+ languages)
    - Fallback 1 : langdetect library (fast, probabilistic)
    - Fallback 2 : Character-based heuristics
    - Confidence scoring
    - Multi-language detection
    - Script detection (Latin, Cyrillic, Arabic, etc.)

    Supported Languages:
    - 100+ languages via XLM-RoBERTa
    - High accuracy for major languages (English, Spanish, French, German, Chinese, etc.)
    """
    # Minimum text length for reliable detection
    MIN_TEXT_LENGTH = 20
    
    # Language name mappings
    LANGUAGE_NAMES  = {"en": "English",
                       "es": "Spanish",
                       "fr": "French",
                       "de": "German",
                       "it": "Italian",
                       "pt": "Portuguese",
                       "ru": "Russian",
                       "zh": "Chinese",
                       "ja": "Japanese",
                       "ko": "Korean",
                       "ar": "Arabic",
                       "hi": "Hindi",
                      }
    
    # Character ranges for script detection
    SCRIPT_RANGES   = {Script.LATIN: [(0x0041, 0x007A), (0x00C0, 0x024F)],
                       Script.CYRILLIC: [(0x0400, 0x04FF)],
                       Script.ARABIC: [(0x0600, 0x06FF), (0x0750, 0x077F)],
                       Script.CHINESE: [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
                       Script.JAPANESE: [(0x3040, 0x309F), (0x30A0, 0x30FF)],
                       Script.KOREAN: [(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
                       Script.DEVANAGARI: [(0x0900, 0x097F)],
                       Script.GREEK: [(0x0370, 0x03FF)],
                       Script.HEBREW: [(0x0590, 0x05FF)],
                       Script.THAI: [(0x0E00, 0x0E7F)],
                      }
    

    def __init__(self, use_model: bool = True, min_confidence: float = 0.5):
        """
        Initialize language detector
        
        Arguments:
        ----------
            use_model       : Use ML model for detection (more accurate)

            min_confidence  : Minimum confidence threshold
        """
        self.use_model      = use_model and MODEL_MANAGER_AVAILABLE
        self.min_confidence = min_confidence
        self.model_manager  = None
        self.classifier     = None
        self.is_initialized = False
        
        logger.info(f"LanguageDetector initialized (use_model={self.use_model})")
    

    def initialize(self) -> bool:
        """
        Initialize the ML model (if using)
        
        Returns:
        --------
            { bool } : True if successful, False otherwise
        """
        if not self.use_model:
            self.is_initialized = True
            return True
        
        try:
            logger.info("Initializing language detection model...")
            
            self.model_manager  = get_model_manager()
            self.classifier     = self.model_manager.load_pipeline(model_name = "language_detector",
                                                                   task       = "text-classification",
                                                                  )
            
            self.is_initialized = True
            logger.success("Language detector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize language detector: {repr(e)}")
            logger.warning("Falling back to langdetect library")
            self.use_model      = False
            self.is_initialized = True
            return False
    

    def detect(self, text: str, **kwargs) -> LanguageDetectionResult:
        """
        Detect language of input text
        
        Arguments:
        ----------
            text     { str } : Input text to analyze

            **kwargs         : Additional options
            
        Returns:
        --------
            LanguageDetectionResult object
        """
        warnings = list()
        
        # Validate input
        if not text or not isinstance(text, str):
            return self._create_unknown_result(text     = "",
                                               warnings = ["Empty or invalid text"],
                                              )
        
        # Clean text for analysis
        cleaned_text = self._clean_text(text)
        char_count   = len(cleaned_text)
        word_count   = len(cleaned_text.split())
        
        # Check minimum length
        if (char_count < self.MIN_TEXT_LENGTH):
            warnings.append(f"Text too short ({char_count} chars, minimum {self.MIN_TEXT_LENGTH}). Detection may be unreliable.")
        
        # Detect script first
        script = self._detect_script(cleaned_text)
        
        # Try detection methods in order
        result = None
        
        # Method 1 : ML Model
        if self.use_model and self.is_initialized:
            try:
                result                  = self._detect_with_model(text = cleaned_text)
                result.detection_method = "xlm-roberta-model"
            
            except Exception as e:
                logger.warning(f"Model detection failed: {repr(e)}, trying fallback")
                warnings.append("Model detection failed, using fallback")
        
        # Method 2 : langdetect library
        if result is None and LANGDETECT_AVAILABLE:
            try:
                result                  = self._detect_with_langdetect(text = cleaned_text)
                result.detection_method = "langdetect-library"
            
            except Exception as e:
                logger.warning(f"langdetect failed: {repr(e)}, trying heuristics")
                warnings.append("langdetect failed, using heuristics")
        
        # Method 3 : Character-based heuristics
        if result is None:
            result                  = self._detect_with_heuristics(cleaned_text, script)
            result.detection_method = "character-heuristics"
        
        # Add metadata
        result.script     = script
        result.char_count = char_count
        result.word_count = word_count

        result.warnings.extend(warnings)
        
        # Check for multilingual content
        if len([v for v in result.all_languages.values() if v > 0.2]) > 1:
            result.is_multilingual = True
            warnings.append("Text appears to contain multiple languages")
        
        logger.info(f"Detected language: {result.primary_language.value} (confidence: {result.confidence:.2f}, method: {result.detection_method})")
        
        return result
    

    def _detect_with_model(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using XLM-RoBERTa model with sentence-based chunking for more accurate detection on long texts
        """
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("Model not initialized")
        
        try:
            # Strategy: Use multiple text chunks for better accuracy
            chunks            = self._split_text_into_chunks(text = text)
            logger.info(f"Split text into {len(chunks)} chunks for language detection")
            
            all_chunk_results = list()
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_result = self._process_single_chunk(chunk = chunk)
                    all_chunk_results.append(chunk_result)
                    
                except Exception as e:
                    logger.warning(f"Chunk {i+1} processing failed: {repr(e)}")
                    continue
            
            if not all_chunk_results:
                raise RuntimeError("All chunks failed processing")
            
            # Aggregate results from all chunks
            return self._aggregate_chunk_results(chunk_results = all_chunk_results)
            
        except Exception as e:
            logger.error(f"Chunk-based model detection failed: {repr(e)}")
            raise


    def _split_text_into_chunks(self, text: str, max_chunk_length: int = 500, min_chunk_length: int = 50) -> List[str]:
        """
        Split text into meaningful chunks for language detection
        
        Arguments:
        ----------
            text             { str } : Input text

            max_chunk_length { int } : Maximum characters per chunk
            
            min_chunk_length { int } : Minimum characters per chunk
            
        Returns:
        --------
            List of text chunks
        """
        if (len(text) <= max_chunk_length):
            return [text]
        
        # Strategy 1: Split by sentences first
        sentences     = re.split(r'[.!?]+', text)
        sentences     = [s.strip() for s in sentences if s.strip()]
        
        chunks        = list()

        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence doesn't exceed max length
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                if current_chunk:
                    current_chunk += " " + sentence
                
                else:
                    current_chunk = sentence
            
            else:
                # Current chunk is full, save it
                if current_chunk and len(current_chunk) >= min_chunk_length:
                    chunks.append(current_chunk)
                
                # Start new chunk with current sentence
                current_chunk = sentence
        
        # Add the last chunk if it meets minimum length
        if (current_chunk and (len(current_chunk) >= min_chunk_length)):
            chunks.append(current_chunk)
        
        # Strategy 2: If sentence splitting didn't work well, use fixed-length chunks
        if ((len(chunks) == 0) or ((len(chunks) == 1 )and (len(chunks[0]) > max_chunk_length))):
            chunks = self._split_fixed_length(text, max_chunk_length)
        
        logger.debug(f"Split {len(text)} chars into {len(chunks)} chunks: {[len(c) for c in chunks]}")
        return chunks


    def _split_fixed_length(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Fallback: Split text into fixed-length chunks
        """
        chunks = list()

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            # Try to break at word boundaries
            if ((i + chunk_size) < len(text)):
                last_space = chunk.rfind(' ')
                # If we found a space in the last 30%
                if (last_space > chunk_size * 0.7):  
                    chunk = chunk[:last_space].strip()

            chunks.append(chunk)
        return chunks


    def _process_single_chunk(self, chunk: str) -> Dict:
        """
        Process a single chunk through the language detection model
        """
        # Get the tokenizer from the pipeline
        tokenizer = self.classifier.tokenizer
        
        # Tokenize with explicit length limits
        inputs    = tokenizer(chunk,
                              return_tensors     = "pt",
                              truncation         = True,
                              max_length         = 512,
                              padding            = True,
                              add_special_tokens = True,
                             )
        
        # Get model from pipeline
        model     = self.classifier.model
        device    = next(model.parameters()).device
        
        # Move inputs to correct device
        inputs    = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs     = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get top predictions for this chunk
        top_predictions = torch.topk(predictions[0], k = 3)
        
        chunk_results   = dict()

        for i in range(len(top_predictions.indices)):
            lang_idx  = top_predictions.indices[i].item()
            score     = top_predictions.values[i].item()
            
            # Get language label from model config
            lang_code = model.config.id2label[lang_idx]
            
            # Handle model output format
            if ('_' in lang_code):
                lang_code = lang_code.split('_')[0]
            
            chunk_results[lang_code] = score
        
        return chunk_results


    def _aggregate_chunk_results(self, chunk_results: List[Dict]) -> LanguageDetectionResult:
        """
        Aggregate results from multiple chunks using weighted averaging
        """
        # Combine scores from all chunks
        all_scores    = dict()
        chunk_weights = list()
        
        for chunk_result in chunk_results:
            # Calculate chunk weight based on confidence and diversity
            top_score    = max(chunk_result.values()) if chunk_result else 0
            # Weight by confidence
            chunk_weight = top_score  
            
            chunk_weights.append(chunk_weight)
            
            for lang_code, score in chunk_result.items():
                if lang_code not in all_scores:
                    all_scores[lang_code] = list()

                all_scores[lang_code].append(score)
        
        # Calculate weighted average for each language
        weighted_scores = dict()

        for lang_code, scores in all_scores.items():
            if (len(scores) != len(chunk_weights)):
                # Use simple average if weight mismatch
                weighted_scores[lang_code] = sum(scores) / len(scores)

            else:
                # Weighted average
                weighted_sum               = sum(score * weight for score, weight in zip(scores, chunk_weights))
                total_weight               = sum(chunk_weights)
                weighted_scores[lang_code] = weighted_sum / total_weight if total_weight > 0 else sum(scores) / len(scores)
        
        # Find primary language
        primary_lang = None
        primary_conf = 0.0
        
        for lang_code, score in weighted_scores.items():
            if (score > primary_conf):
                primary_conf = score
                primary_lang = lang_code
        
        # Convert to Language enum
        try:
            primary_language = Language(primary_lang)

        except ValueError:
            primary_language = Language.UNKNOWN
        
        # Calculate detection quality metrics
        detection_quality = self._assess_detection_quality(chunk_results, weighted_scores)
        
        warnings          = list()

        if detection_quality.get('low_confidence', False):
            warnings.append("Low confidence across multiple chunks")

        if detection_quality.get('inconsistent', False):
            warnings.append("Inconsistent language detection across chunks")
        
        return LanguageDetectionResult(primary_language = primary_language,
                                       confidence       = primary_conf,
                                       all_languages    = weighted_scores,
                                       script           = Script.UNKNOWN,
                                       is_multilingual  = detection_quality.get('multilingual', False),
                                       detection_method = "model-chunked",
                                       char_count       = 0,
                                       word_count       = 0,
                                       warnings         = warnings,
                                      )


    def _assess_detection_quality(self, chunk_results: List[Dict], final_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Assess the quality and consistency of language detection across chunks
        """
        quality_metrics = {'low_confidence' : False,
                           'inconsistent'   : False,
                           'multilingual'   : False,
                          }
        
        if not chunk_results:
            return quality_metrics
        
        # Check for low confidence
        avg_top_confidence = sum(max(chunk.values()) for chunk in chunk_results) / len(chunk_results)
        if (avg_top_confidence < 0.6):
            quality_metrics['low_confidence'] = True
        
        # Check for inconsistency (different primary languages across chunks)
        chunk_primaries = list()

        for chunk in chunk_results:
            if chunk:
                primary = max(chunk.items(), key=lambda x: x[1])[0]
                chunk_primaries.append(primary)
        
        if (len(set(chunk_primaries)) > 1):
            quality_metrics['inconsistent'] = True
        
        # Check for multilingual content
        strong_languages = [lang for lang, score in final_scores.items() if score > 0.2]
        if (len(strong_languages) > 1):
            quality_metrics['multilingual'] = True
        
        return quality_metrics

    
    def _detect_with_langdetect(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using langdetect library
        """
        # Get all language probabilities
        lang_probs    = detect_langs(text)
        
        all_languages = dict()

        for prob in lang_probs:
            all_languages[prob.lang] = prob.prob
        
        # Primary language
        primary       = lang_probs[0]
        
        try:
            primary_language = Language(primary.lang)

        except ValueError:
            primary_language = Language.UNKNOWN
        
        return LanguageDetectionResult(primary_language = primary_language,
                                       confidence       = primary.prob,
                                       all_languages    = all_languages,
                                       script           = Script.UNKNOWN,
                                       is_multilingual  = False,
                                       detection_method = "langdetect",
                                       char_count       = 0,
                                       word_count       = 0,
                                       warnings         = [],
                                      )
    

    def _detect_with_heuristics(self, text: str, script: Script) -> LanguageDetectionResult:
        """
        Detect language using character-based heuristics
        """
        # Script-based language mapping
        script_to_language = {Script.CHINESE    : Language.CHINESE,
                              Script.JAPANESE   : Language.JAPANESE,
                              Script.KOREAN     : Language.KOREAN,
                              Script.ARABIC     : Language.ARABIC,
                              Script.CYRILLIC   : Language.RUSSIAN,
                              Script.DEVANAGARI : Language.HINDI,
                              Script.GREEK      : Language.GREEK,
                              Script.HEBREW     : Language.HEBREW,
                              Script.THAI       : Language.THAI,
                             }
        
        # If script clearly indicates language
        if script in script_to_language:
            primary_language = script_to_language[script]
            # Moderate confidence for heuristics
            confidence       = 0.7  

        else:
            # For Latin script, check common words
            primary_language = self._detect_latin_language(text)
            # Lower confidence
            confidence       = 0.5  
        
        return LanguageDetectionResult(primary_language = primary_language,
                                       confidence       = confidence,
                                       all_languages    = {primary_language.value: confidence},
                                       script           = script,
                                       is_multilingual  = False,
                                       detection_method = "heuristics",
                                       char_count       = 0,
                                       word_count       = 0,
                                       warnings         = ["Detection using heuristics, accuracy may be limited"],
                                      )
    

    def _detect_latin_language(self, text: str) -> Language:
        """
        Detect Latin-script language using common word patterns
        """
        text_lower = text.lower()
        
        # Common word patterns for major Latin-script languages
        patterns   = {Language.ENGLISH    : ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'on', 'this', 'are', 'was', 'be', 'have', 'from', 'or', 'by'],
                      Language.SPANISH    : ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'por', 'con', 'no', 'una', 'para', 'es', 'al', 'como', 'del', 'los', 'se', 'las', 'su'],
                      Language.FRENCH     : ['le', 'de', 'un', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son', 'que', 'ce', 'du', 'quel', 'elle', 'dans', 'pour', 'au', 'avec'],
                      Language.GERMAN     : ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als'],
                      Language.ITALIAN    : ['di', 'e', 'il', 'la', 'che', 'per', 'un', 'in', 'è', 'a', 'non', 'una', 'da', 'sono', 'come', 'del', 'ma', 'si', 'nel', 'anche'],
                      Language.PORTUGUESE : ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais'],
                     }
        
        # Count matches for each language
        scores     = dict()
        words      = set(text_lower.split())
        
        for lang, common_words in patterns.items():
            score        = sum(1 for word in common_words if word in words)
            scores[lang] = score
        
        # Return language with highest score
        if scores:
            best_lang = max(scores.items(), key = lambda x: x[1])
            # At least 3 matches
            if (best_lang[1] > 2):  
                return best_lang[0]
        
        # Default to English for Latin script
        return Language.ENGLISH
    

    def _detect_script(self, text: str) -> Script:
        """
        Detect the writing script used in text
        """
        # Count characters in each script
        script_counts = {script: 0 for script in Script if script not in [Script.MIXED, Script.UNKNOWN]}
        
        for char in text:
            if char in string.whitespace or char in string.punctuation:
                continue
            
            code_point = ord(char)
            
            for script, ranges in self.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if (start <= code_point <= end):
                        script_counts[script] += 1
                        break
        
        # Find dominant script
        total_chars = sum(script_counts.values())
        
        if (total_chars == 0):
            return Script.UNKNOWN
        
        # Calculate percentages
        script_percentages = {script: count / total_chars for script, count in script_counts.items() if count > 0}
        
        # Check if mixed (no single script > 70%)
        if (len(script_percentages) > 1):
            max_percentage = max(script_percentages.values())
            if (max_percentage < 0.7):
                return Script.MIXED
        
        # Return dominant script
        if script_percentages:
            return max(script_percentages.items(), key=lambda x: x[1])[0]
        
        return Script.UNKNOWN
    

    def _clean_text(self, text: str) -> str:
        """
        Clean text for language detection
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    

    def _create_unknown_result(self, text: str, warnings: List[str]) -> LanguageDetectionResult:
        """
        Create result for unknown language
        """
        return LanguageDetectionResult(primary_language = Language.UNKNOWN,
                                       confidence       = 0.0,
                                       all_languages    = {},
                                       script           = Script.UNKNOWN,
                                       is_multilingual  = False,
                                       detection_method = "none",
                                       char_count       = len(text),
                                       word_count       = len(text.split()),
                                       warnings         = warnings,
                                      )

    
    def is_language(self, text: str, target_language: Language, threshold: float = 0.7) -> bool:
        """
        Check if text is in a specific language
        
        Arguments:
        ----------
            text            : Input text

            target_language : Language to check for
            
            threshold       : Minimum confidence threshold
            
        Returns:
        --------
            { bool }        : True if text is in target language with sufficient confidence
        """
        result = self.detect(text)
        return ((result.primary_language == target_language) and (result.confidence >= threshold))

    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        """
        return [lang.value for lang in Language if (lang != Language.UNKNOWN)]
    

    def cleanup(self):
        """
        Clean up resources
        """
        self.classifier     = None
        self.is_initialized = False


# Convenience Functions 
def quick_detect(text: str, **kwargs) -> LanguageDetectionResult:
    """
    Quick language detection with default settings
    
    Arguments:
    ----------
        text     : Input text

        **kwargs : Override settings
        
    Returns:
    --------
        LanguageDetectionResult object
    """
    detector = LanguageDetector(**kwargs)
    
    if detector.use_model:
        detector.initialize()

    return detector.detect(text)


def is_english(text: str, threshold: float = 0.7) -> bool:
    """
    Quick check if text is English
    """
    detector   = LanguageDetector(use_model = True)
    is_english = detector.is_language(text, Language.ENGLISH, threshold)
    
    return is_english



# Export
__all__ = ['Script',
           'Language',
           'is_english',
           'quick_detect',
           'LanguageDetector',
           'LanguageDetectionResult',
          ]
