# DEPENDENCIES
import re
import unicodedata
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from loguru import logger
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """
    Container for processed text with metadata
    """
    original_text      : str
    cleaned_text       : str
    sentences          : List[str]
    words              : List[str]
    paragraphs         : List[str]
    char_count         : int
    word_count         : int
    sentence_count     : int
    paragraph_count    : int
    avg_sentence_length: float
    avg_word_length    : float
    is_valid           : bool
    validation_errors  : List[str]
    metadata           : Dict[str, Any]
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization
        """
        return {"original_length"      : len(self.original_text),
                "cleaned_length"       : len(self.cleaned_text),
                "char_count"           : self.char_count,
                "word_count"           : self.word_count,
                "sentence_count"       : self.sentence_count,
                "paragraph_count"      : self.paragraph_count,
                "avg_sentence_length"  : round(self.avg_sentence_length, 2),
                "avg_word_length"      : round(self.avg_word_length, 2),
                "is_valid"             : self.is_valid,
                "validation_errors"    : self.validation_errors,
                "metadata"             : self.metadata,
               }


class TextProcessor:
    """
    Handles text cleaning, normalization, sentence splitting, and preprocessing for AI detection metrics
    
    Features::
    - Unicode normalization
    - Smart sentence splitting (handles abbreviations, decimals, etc.)
    - Whitespace normalization
    - Special character handling
    - Paragraph detection
    - Word tokenization
    - Text validation
    - Chunk creation for long texts
    """
    
    # Common abbreviations that shouldn't trigger sentence breaks
    ABBREVIATIONS     = {'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 'ph.d', 'inc', 'ltd', 'corp', 'co', 'vs', 'etc', 'e.g', 'i.e', 'al', 'fig', 'vol', 'no', 'approx', 'est', 'min', 'max', 'avg', 'dept', 'assoc', 'bros', 'u.s', 'u.k', 'a.m', 'p.m', 'b.c', 'a.d', 'st', 'ave', 'blvd'}
    
    # Patterns for sentence splitting
    SENTENCE_ENDINGS  = r'[.!?]+(?=\s+[A-Z]|$)'
    
    # Patterns for cleaning
    MULTIPLE_SPACES   = re.compile(r'\s+')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    

    def __init__(self, min_text_length: int = 50, max_text_length: int = 500000, preserve_formatting: bool = False, remove_urls: bool = True, remove_emails: bool = True,
                 normalize_unicode: bool = True, fix_encoding: bool = True):
        """
        Initialize text processor
        
        Arguments:
        ----------
            min_text_length      : Minimum acceptable text length

            max_text_length      : Maximum text length to process
            
            preserve_formatting  : Keep original line breaks and spacing
            
            remove_urls          : Remove URLs from text
            
            remove_emails        : Remove email addresses
            
            normalize_unicode    : Normalize Unicode characters
            
            fix_encoding         : Fix common encoding issues
        """
        self.min_text_length     = min_text_length
        self.max_text_length     = max_text_length
        self.preserve_formatting = preserve_formatting
        self.remove_urls         = remove_urls
        self.remove_emails       = remove_emails
        self.normalize_unicode   = normalize_unicode
        self.fix_encoding        = fix_encoding
        
        logger.info(f"TextProcessor initialized with min_length={min_text_length}, max_length={max_text_length}")
    

    def process(self, text: str, **kwargs) -> ProcessedText:
        """
        Main processing pipeline
        
        Arguments:
        ----------
            text     { str }  : Input text to process

            **kwargs          : Override default settings
            
        Returns:
        --------
            { ProcessedText } : ProcessedText object with all processed components
        """
        try:
            original_text     = text
            validation_errors = list()
            
            # Validate input
            if not text or not isinstance(text, str):
                validation_errors.append("Text is empty or not a string")
                return self._create_invalid_result(original_text, validation_errors)
            
            # Initial cleaning
            text = self._initial_clean(text)
            
            # Fix encoding issues
            if self.fix_encoding:
                text = self._fix_encoding_issues(text)
            
            # Normalize Unicode
            if self.normalize_unicode:
                text = self._normalize_unicode(text)
            
            # Remove unwanted elements
            if self.remove_urls:
                text = self._remove_urls(text)
            
            if self.remove_emails:
                text = self._remove_emails(text)
            
            # Clean whitespace
            text = self._clean_whitespace(text)
            
            # Validate length
            if (len(text) < self.min_text_length):
                validation_errors.append(f"Text too short: {len(text)} chars (minimum: {self.min_text_length})")
            
            if (len(text) > self.max_text_length):
                validation_errors.append(f"Text too long: {len(text)} chars (maximum: {self.max_text_length})")
                text = text[:self.max_text_length]
            
            # Extract components
            sentences    = self.split_sentences(text)
            words        = self.tokenize_words(text)
            paragraphs   = self.split_paragraphs(text)
            
            # Calculate statistics
            char_count   = len(text)
            word_count   = len(words)
            sent_count   = len(sentences)
            para_count   = len(paragraphs)
            
            avg_sent_len = word_count / sent_count if sent_count > 0 else 0
            avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
            
            # Additional validation
            if (sent_count == 0):
                validation_errors.append("No valid sentences found")
            
            if (word_count < 10):
                validation_errors.append(f"Too few words: {word_count} (minimum: 10)")
            
            # Create metadata
            metadata = {"has_special_chars" : self._has_special_characters(text),
                        "has_numbers"       : any(c.isdigit() for c in text),
                        "has_uppercase"     : any(c.isupper() for c in text),
                        "has_lowercase"     : any(c.islower() for c in text),
                        "unique_words"      : len(set(w.lower() for w in words)),
                        "lexical_diversity" : len(set(w.lower() for w in words)) / word_count if word_count > 0 else 0,
                       }
            
            is_valid = len(validation_errors) == 0
            
            return ProcessedText(original_text       = original_text,
                                 cleaned_text        = text,
                                 sentences           = sentences,
                                 words               = words,
                                 paragraphs          = paragraphs,
                                 char_count          = char_count,
                                 word_count          = word_count,
                                 sentence_count      = sent_count,
                                 paragraph_count     = para_count,
                                 avg_sentence_length = avg_sent_len,
                                 avg_word_length     = avg_word_len,
                                 is_valid            = is_valid,
                                 validation_errors   = validation_errors,
                                 metadata            = metadata,
                                )
            
        except Exception as e:
            logger.error(f"Error processing text: {repr(e)}")
            return self._create_invalid_result(text if text else "", [f"Processing error: {str(e)}"])
    

    def split_sentences(self, text: str) -> List[str]:
        """
        Smart sentence splitting with abbreviation handling
        
        Arguments:
        ----------
            text { str } : Input text
            
        Returns:
        --------
             { list}     : List of sentences
        """
        # Protect abbreviations
        protected_text = text

        for abbr in self.ABBREVIATIONS:
            # Replace abbreviation periods with placeholder
            protected_text = re.sub(pattern = rf'\b{re.escape(abbr)}\.',
                                    repl    = abbr.replace('.', '<DOT>'),
                                    string  = protected_text,
                                    flags   = re.IGNORECASE,
                                   )
        
        # Protect decimal numbers (e.g., 3.14)
        protected_text    = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', protected_text)
        
        # Protect ellipsis
        protected_text    = protected_text.replace('...', '<ELLIPSIS>')
        
        # Split on sentence endings
        sentences         = re.split(self.SENTENCE_ENDINGS, protected_text)
        
        # Restore protected characters and clean
        cleaned_sentences = list()

        for sent in sentences:
            sent = sent.replace('<DOT>', '.')
            sent = sent.replace('<ELLIPSIS>', '...')
            sent = sent.strip()
            
            # Only keep non-empty sentences with actual words
            if (sent and (len(sent.split()) >= 2)):  
                # At least 2 words
                cleaned_sentences.append(sent)
        
        return cleaned_sentences

    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Arguments:
        ----------
            text { str } : Input text
            
        Returns:
        --------
             { list }    : List of words
        """
        # Remove punctuation but keep apostrophes in contractions
        text           = re.sub(pattern = r"[^\w\s'-]", 
                                repl    = ' ', 
                                string  = text,
                               )
        
        # Split on whitespace
        words          = text.split()
        
        # Filter out pure numbers and single characters (except 'a' and 'I')
        filtered_words = list()

        for word in words:
            # Remove leading/trailing quotes and hyphens
            word = word.strip("'-")  
            if word and (len(word) > 1 or word.lower() in ['a', 'i']):
                if not word.replace('-', '').replace("'", '').isdigit():
                    filtered_words.append(word)
        
        return filtered_words

    
    def split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs
        
        Arguments:
        ----------
            text { str } : Input text
            
        Returns:
        --------
            { list }     : List of paragraphs
        """
        # Split on double newlines or more
        paragraphs         = re.split(r'\n\s*\n', text)
        
        # Clean and filter
        cleaned_paragraphs = list()

        for para in paragraphs:
            para = para.strip()
            
            # There should be at least 5 words
            if para and (len(para.split()) >= 5):  
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs if cleaned_paragraphs else [text]
    

    def create_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50, unit: str = 'words') -> List[str]:
        """
        Split long text into overlapping chunks
        
        Arguments:
        ----------
            text        { str } : Input text

            chunk_size  { int } : Size of each chunk
            
            overlap     { int } : Number of units to overlap between chunks
            
            unit        { str } : 'words', 'sentences', or 'chars'
            
        Returns:
        --------
                { list }        : List of text chunks
        """
        if (unit == 'words'):
            units = self.tokenize_words(text)

        elif (unit == 'sentences'):
            units = self.split_sentences(text)

        elif (unit == 'chars'):
            units = list(text)

        else:
            raise ValueError(f"Unknown unit: {unit}")
        
        if (len(units) <= chunk_size):
            return [text]
        
        chunks = list()
        start  = 0
        
        while (start < len(units)):
            end         = start + chunk_size
            chunk_units = units[start:end]
            
            if (unit == 'chars'):
                chunk_text = ''.join(chunk_units)

            else:
                chunk_text = ' '.join(chunk_units)
            
            chunks.append(chunk_text)
            start = end - overlap
        
        return chunks
    

    def _initial_clean(self, text: str) -> str:
        """
        Remove null bytes and control characters
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove other control characters except newlines and tabs
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t\r')
        
        return text

    
    def _fix_encoding_issues(self, text: str) -> str:
        """
        Fix common encoding issues
        """
        replacements = {'â€™' : "'",   # Smart apostrophe
                        'â€œ' : '"',   # Smart quote left
                        'â€'  : '"',   # Smart quote right
                        'â€"' : '—',   # Em dash
                        'â€"' : '–',   # En dash
                        'â€¦' : '...', # Ellipsis
                        'Ã©'  : 'é',   # Common UTF-8 issue
                        'Ã¨'  : 'è',
                        'Ã '  : 'à',
                        'â‚¬' : '€',   # Euro sign
                       }
        
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        return text

    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode to consistent form
        """
        # NFKC normalization (compatibility decomposition, followed by canonical composition)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace smart quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')
        
        return text

    
    def _remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        """
        # Remove http/https URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove www URLs
        text = re.sub(r'www\.\S+', '', text)
        
        return text
    

    def _remove_emails(self, text: str) -> str:
        """
        Remove email addresses
        """
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        return text
    

    def _clean_whitespace(self, text: str) -> str:
        """
        Normalize whitespace
        """
        if self.preserve_formatting:
            # Just normalize multiple spaces
            text = self.MULTIPLE_SPACES.sub(' ', text)
            text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
        
        else:
            # Aggressive whitespace normalization
            text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
            text = self.MULTIPLE_SPACES.sub(' ', text)
            text = text.strip()
        
        return text
    

    def _has_special_characters(self, text: str) -> bool:
        """
        Check if text contains special characters
        """
        special_chars = set('!@#$%^&*()[]{}|\\:;"<>?,./~`')
        return any(char in special_chars for char in text)
    

    def _create_invalid_result(self, text: str, errors: List[str]) -> ProcessedText:
        """
        Create a ProcessedText object for invalid input
        """
        return ProcessedText(original_text       = text,
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
                             validation_errors   = errors,
                             metadata            = {},
                            )



# Convenience Functions

def quick_process(text: str, **kwargs) -> ProcessedText:
    """
    Quick processing with default settings
    
    Arguments:
    ----------
        text     : Input text

        **kwargs : Override settings
        
    Returns:
    --------
        ProcessedText object
    """
    processor = TextProcessor(**kwargs)
    return processor.process(text)


def extract_sentences(text: str) -> List[str]:
    """
    Quick sentence extraction
    """
    processor = TextProcessor()
    return processor.split_sentences(text)


def extract_words(text: str) -> List[str]:
    """
    Quick word extraction
    """
    processor = TextProcessor()
    return processor.tokenize_words(text)


# Export
__all__ = ['TextProcessor',
           'ProcessedText',
           'quick_process',
           'extract_sentences',
           'extract_words',
          ]


# ==================== Testing ====================
if __name__ == "__main__":
    # Test cases
    test_texts = [
        # Normal text
        "This is a test. Dr. Smith works at the U.S. Department of Education. "
        "He published a paper on AI detection in 2024.",
        
        # Text with encoding issues
        "This textâ€™s got some â€œweirdâ€ characters that need fixing.",
        
        # Text with URLs and emails
        "Check out https://example.com or email me at test@example.com for more info.",
        
        # Short text (should fail validation)
        "Too short.",
        
        # Text with numbers and special characters
        "The price is $19.99 for version 2.0. Contact us at (555) 123-4567!",
    ]
    
    processor = TextProcessor(min_text_length=20)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"{'='*70}")
        print(f"Input: {text[:100]}...")
        
        result = processor.process(text)
        
        print(f"\nValid: {result.is_valid}")
        if not result.is_valid:
            print(f"Errors: {result.validation_errors}")
        
        print(f"Word count: {result.word_count}")
        print(f"Sentence count: {result.sentence_count}")
        print(f"Avg sentence length: {result.avg_sentence_length:.2f}")
        print(f"\nSentences:")
        for j, sent in enumerate(result.sentences[:3], 1):
            print(f"  {j}. {sent}")