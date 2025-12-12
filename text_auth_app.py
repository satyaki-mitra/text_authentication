# DEPENDENCIES
import os
import time
import json
import uvicorn
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from typing import Union
from pathlib import Path
from fastapi import File
from fastapi import Form
from loguru import logger
from pydantic import Field
from typing import Optional
from fastapi import FastAPI
from fastapi import Request
from datetime import datetime
from fastapi import UploadFile
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi import BackgroundTasks
from config.settings import settings
from utils.logger import central_logger
from utils.logger import log_api_request
from detector.attribution import AIModel
from config.threshold_config import Domain
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from utils.logger import log_detection_event
from detector.attribution import ModelAttributor
from detector.highlighter import TextHighlighter
from processors.language_detector import Language
from detector.orchestrator import DetectionResult
from detector.attribution import AttributionResult
from fastapi.middleware.cors import CORSMiddleware
from processors.text_processor import TextProcessor
from reporter.report_generator import ReportGenerator
from detector.orchestrator import DetectionOrchestrator
from processors.domain_classifier import DomainClassifier
from processors.language_detector import LanguageDetector
from processors.document_extractor import DocumentExtractor
from reporter.reasoning_generator import ReasoningGenerator



# ==================== CUSTOM SERIALIZATION ====================
class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types and custom objects
    """
    def default(self, obj: Any) -> Any:
        """
        Convert non-serializable objects to JSON-serializable types
        """
        # NumPy types
        if (isinstance(obj, (np.float32, np.float64))):
            return float(obj)

        elif (isinstance(obj, (np.int32, np.int64, np.int8, np.uint8))):
            return int(obj)

        elif (isinstance(obj, np.ndarray)):
            return obj.tolist()

        elif (isinstance(obj, np.bool_)):
            return bool(obj)

        elif (hasattr(obj, 'item')):  
            # numpy scalar types
            return obj.item()
        
        # Custom objects with to_dict method
        elif (hasattr(obj, 'to_dict')):
            return obj.to_dict()
        
        # Pydantic models
        elif (hasattr(obj, 'dict')):
            return obj.dict()
        
        # Handle other types
        elif (isinstance(obj, (set, tuple))):
            return list(obj)
        
        return super().default(obj)


class NumpyJSONResponse(JSONResponse):
    """
    Custom JSON response that handles NumPy types
    """
    def render(self, content: Any) -> bytes:
        """
        Render content with NumPy type handling
        """
        return json.dumps(obj          = content,
                          ensure_ascii = False,
                          allow_nan    = False,
                          indent       = None,
                          separators   = (",", ":"),
                          cls          = NumpyJSONEncoder,
                         ).encode("utf-8")


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types
    
    Arguments:
    ----------
        obj : Any Python object that may contain NumPy types
        
    Returns:
    --------
        Object with all NumPy types converted to native Python types
    """
    if (obj is None):
        return None
    
    # Handle dictionaries
    if (isinstance(obj, dict)):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    
    # Handle lists, tuples, sets
    elif (isinstance(obj, (list, tuple, set))):
        return [convert_numpy_types(item) for item in obj]
    
    # Handle NumPy types
    elif (isinstance(obj, (np.float32, np.float64))):
        return float(obj)
    
    elif (isinstance(obj, (np.int32, np.int64, np.int8, np.uint8))):
        return int(obj)
    
    elif (isinstance(obj, np.ndarray)):
        return obj.tolist()
    
    elif (isinstance(obj, np.bool_)):
        return bool(obj)
    
    # numpy scalar types
    elif (hasattr(obj, 'item')):  
        return obj.item()
    
    # Handle custom objects with to_dict method
    elif (hasattr(obj, 'to_dict')):
        return convert_numpy_types(obj.to_dict())
    
    # Handle Pydantic models
    elif (hasattr(obj, 'dict')):
        return convert_numpy_types(obj.dict())
    
    # Return as-is for other types (str, int, float, bool, etc.)
    else:
        return obj


def safe_serialize_response(data: Any) -> Any:
    """
    Safely serialize response data ensuring all types are JSON-compatible
    
    Arguments:
    ----------
        data : Response data to serialize
        
    Returns:
    --------
        Fully serializable data structure
    """
    return convert_numpy_types(data)


# ==================== PYDANTIC DATACLASS MODELS ====================
class SerializableBaseModel(BaseModel):
    """
    Base model with enhanced serialization for NumPy types
    """
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Override dict method to handle NumPy types
        """
        data = super().dict(*args, **kwargs)
        
        return convert_numpy_types(data)
    

    def json(self, *args, **kwargs) -> str:
        """
        Override json method to handle NumPy types
        """
        data = self.dict(*args, **kwargs)
        
        return json.dumps(data, cls=NumpyJSONEncoder, *args, **kwargs)


class TextAnalysisRequest(SerializableBaseModel):
    """
    Request model for text analysis
    """
    text                    : str           = Field(..., min_length = 50, max_length = 50000, description = "Text to analyze")
    domain                  : Optional[str] = Field(None, description = "Override automatic domain detection")
    enable_attribution      : bool          = Field(True, description = "Enable AI model attribution")
    enable_highlighting     : bool          = Field(True, description = "Generate sentence highlighting")
    skip_expensive_metrics  : bool          = Field(False, description = "Skip computationally expensive metrics")
    use_sentence_level      : bool          = Field(True, description = "Use sentence-level analysis for highlighting") 
    include_metrics_summary : bool          = Field(True, description = "Include metrics summary in highlights")
    generate_report         : bool          = Field(False, description = "Generate detailed PDF/JSON report")


class TextAnalysisResponse(SerializableBaseModel):
    """
    Response model for text analysis
    """
    status           : str
    analysis_id      : str
    detection_result : Dict[str, Any]
    attribution      : Optional[Dict[str, Any]] = None
    highlighted_html : Optional[str]            = None
    reasoning        : Optional[Dict[str, Any]] = None
    report_files     : Optional[Dict[str, str]] = None
    processing_time  : float
    timestamp        : str


class BatchAnalysisRequest(SerializableBaseModel):
    """
    Request model for batch analysis
    """
    texts                  : List[str]     = Field(..., min_items = 1, max_items = 100)
    domain                 : Optional[str] = None
    enable_attribution     : bool          = False
    skip_expensive_metrics : bool          = True
    generate_reports       : bool          = False


class BatchAnalysisResult(SerializableBaseModel):
    """
    Individual batch analysis result
    """
    index        : int
    status       : str
    detection    : Optional[Dict[str, Any]] = None
    attribution  : Optional[Dict[str, Any]] = None
    reasoning    : Optional[Dict[str, Any]] = None
    report_files : Optional[Dict[str, str]] = None
    error        : Optional[str]            = None


class BatchAnalysisResponse(SerializableBaseModel):
    """
    Batch analysis response
    """
    status          : str
    batch_id        : str
    total           : int
    successful      : int
    failed          : int
    results         : List[BatchAnalysisResult]
    processing_time : float
    timestamp       : str


class FileAnalysisResponse(SerializableBaseModel):
    """
    File analysis response
    """
    status           : str
    analysis_id      : str
    file_info        : Dict[str, Any]
    detection_result : Dict[str, Any]
    attribution      : Optional[Dict[str, Any]] = None
    highlighted_html : Optional[str]            = None 
    reasoning        : Optional[Dict[str, Any]] = None
    report_files     : Optional[Dict[str, str]] = None
    processing_time  : float
    timestamp        : str


class HealthCheckResponse(SerializableBaseModel):
    """
    Health check response
    """
    status        : str
    version       : str
    uptime        : float
    models_loaded : Dict[str, bool]


class ReportGenerationResponse(SerializableBaseModel):
    """
    Report generation response
    """
    status      : str
    analysis_id : str
    reports     : Dict[str, str]
    timestamp   : str


class ErrorResponse(SerializableBaseModel):
    """
    Error response model
    """
    status    : str
    error     : str
    timestamp : str


# ==================== ANALYSIS CACHE ====================
class AnalysisCache:
    """
    In-memory cache for storing analysis results
    """
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize cache with size limit and TTL
        
        Arguments:
        ----------
            max_size     : Maximum number of cached items
            ttl_seconds  : Time-to-live for cached items in seconds
        """
        self.cache       = {}
        self.max_size    = max_size
        self.ttl_seconds = ttl_seconds
        logger.info(f"AnalysisCache initialized (max_size={max_size}, ttl={ttl_seconds}s)")
    
    def set(self, analysis_id: str, data: Dict[str, Any]) -> None:
        """
        Store analysis result in cache
        """
        # Clean expired entries first
        self._cleanup_expired()
        
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
            logger.debug(f"Cache full, removed oldest entry: {oldest_key}")
        
        # Store new entry
        self.cache[analysis_id] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.debug(f"Cached analysis: {analysis_id} (cache size: {len(self.cache)})")
    
    def get(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve analysis result from cache
        """
        if analysis_id not in self.cache:
            logger.debug(f"Cache miss: {analysis_id}")
            return None
        
        entry = self.cache[analysis_id]
        
        # Check if expired
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            del self.cache[analysis_id]
            logger.debug(f"Cache expired: {analysis_id}")
            return None
        
        logger.debug(f"Cache hit: {analysis_id}")
        return entry['data']
    
    def _cleanup_expired(self) -> None:
        """
        Remove expired entries from cache
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry['timestamp'] > self.ttl_seconds
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear(self) -> None:
        """
        Clear all cached entries
        """
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cache cleared ({count} entries removed)")
    
    def size(self) -> int:
        """
        Get current cache size
        """
        return len(self.cache)


# ==================== FASTAPI APPLICATION ====================
app = FastAPI(title                  = "TEXT-AUTH AI Detection API",
              description            = "API for detecting AI-generated text",
              version                = "1.0.0",
              docs_url               = "/api/docs",
              redoc_url              = "/api/redoc",
              default_response_class = NumpyJSONResponse,
             )

# CORS Configuration
app.add_middleware(CORSMiddleware,
                   allow_origins     = settings.CORS_ORIGINS,
                   allow_credentials = True,
                   allow_methods     = ["*"],
                   allow_headers     = ["*"],
                  )

# Mount static files
ui_static_path = Path(__file__).parent / "ui" / "static"

if ui_static_path.exists():
    app.mount("/static", StaticFiles(directory = str(ui_static_path)), name = "static")


# Global instances
orchestrator       : Optional[DetectionOrchestrator] = None
attributor         : Optional[ModelAttributor]       = None
highlighter        : Optional[TextHighlighter]       = None
reporter           : Optional[ReportGenerator]       = None
reasoning_generator: Optional[ReasoningGenerator]    = None
document_extractor : Optional[DocumentExtractor]     = None
analysis_cache     : Optional[AnalysisCache]         = None 


# App state
app_start_time                                       = time.time()

initialization_status                                = {"orchestrator"        : False,
                                                        "attributor"          : False,
                                                        "highlighter"         : False,
                                                        "reporter"            : False,
                                                        "reasoning_generator" : False,
                                                        "document_extractor"  : False,
                                                        "analysis_cache"      : False, 
                                                        }


# ==================== APPLICATION LIFECYCLE ====================
@app.on_event("startup")
async def startup_event():
    """
    Initialize all components on startup
    """
    global orchestrator
    global attributor
    global highlighter
    global reporter
    global reasoning_generator
    global document_extractor
    global analysis_cache
    global initialization_status

    # Initialize centralized logging first
    if not central_logger.initialize():
        raise RuntimeError("Failed to initialize logging system")
    
    logger.info("=" * 80)
    logger.info("TEXT-AUTH API Starting Up...")
    logger.info("=" * 80)
    
    try:
        # Initialize Detection Orchestrator
        logger.info("Initializing Detection Orchestrator...")
        orchestrator = DetectionOrchestrator(enable_language_detection = True,
                                             parallel_execution        = False,
                                             skip_expensive_metrics    = False,
                                            )
        
        if orchestrator.initialize():
            initialization_status["orchestrator"] = True
            logger.success("✓ Detection Orchestrator initialized")
        
        else:
            logger.warning("⚠ Detection Orchestrator initialization incomplete")
        
        # Initialize Model Attributor
        logger.info("Initializing Model Attributor...")
        
        attributor = ModelAttributor()
        
        if attributor.initialize():
            initialization_status["attributor"] = True
            logger.success("✓ Model Attributor initialized")
        
        else:
            logger.warning("⚠ Model Attributor initialization incomplete")
        
        # Initialize Text Highlighter
        logger.info("Initializing Text Highlighter...")
       
        highlighter                          = TextHighlighter()
        
        initialization_status["highlighter"] = True
        
        logger.success("✓ Text Highlighter initialized")
        
        # Initialize Report Generator
        logger.info("Initializing Report Generator...")
        
        reporter                          = ReportGenerator()
        
        initialization_status["reporter"] = True
        
        logger.success("✓ Report Generator initialized")
        
        # Initialize Reasoning Generator
        logger.info("Initializing Reasoning Generator...")
        
        reasoning_generator                          = ReasoningGenerator()
        
        initialization_status["reasoning_generator"] = True
        
        logger.success("✓ Reasoning Generator initialized")
        
        # Initialize Document Extractor
        logger.info("Initializing Document Extractor...")
        
        document_extractor                          = DocumentExtractor()
        
        initialization_status["document_extractor"] = True
        
        logger.success("✓ Document Extractor initialized")
        
        # Initialize Analysis Cache: 1 hour TTL
        logger.info("Initializing Analysis Cache...")
        
        analysis_cache = AnalysisCache(max_size    = 100, 
                                       ttl_seconds = 3600,
                                      )  
        
        initialization_status["analysis_cache"] = True
        
        logger.success("✓ Analysis Cache initialized")
        
        logger.info("=" * 80)
        logger.success("TEXT-AUTH API Ready!")
        logger.info(f"Server: {settings.HOST}:{settings.PORT}")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Device: {settings.DEVICE}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


# Cleanup in shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    if analysis_cache:
        analysis_cache.clear()
    
    central_logger.cleanup()
    
    logger.info("Shutdown complete")



# ==================== UTILITY FUNCTIONS ====================
def _get_domain_description(domain: Domain) -> str:
    """
    Get description for a domain
    """
    descriptions = {Domain.GENERAL       : "General content without specific domain",
                    Domain.ACADEMIC      : "Academic papers, essays, research",
                    Domain.CREATIVE      : "Creative writing, fiction, poetry",
                    Domain.AI_ML         : "AI/ML research papers, technical content",
                    Domain.SOFTWARE_DEV  : "Software development, code, documentation",
                    Domain.TECHNICAL_DOC : "Technical documentation, manuals, specs",
                    Domain.ENGINEERING   : "Engineering documents, technical reports",
                    Domain.SCIENCE       : "Scientific papers, research articles",
                    Domain.BUSINESS      : "Business documents, reports, proposals",
                    Domain.LEGAL         : "Legal documents, contracts, court filings",
                    Domain.MEDICAL       : "Medical documents, clinical notes, research",
                    Domain.JOURNALISM    : "News articles, journalistic content",
                    Domain.MARKETING     : "Marketing copy, advertisements, campaigns",
                    Domain.SOCIAL_MEDIA  : "Social media posts, blogs, casual writing",
                    Domain.BLOG_PERSONAL : "Personal blogs, diary entries",
                    Domain.TUTORIAL      : "Tutorials, how-to guides, educational content",
                   }

    return descriptions.get(domain, "")


def _parse_domain(domain_str: Optional[str]) -> Optional[Domain]:
    """
    Parse domain string to Domain enum with comprehensive alias support
    """
    if not domain_str:
        return None
    
    # First try exact match
    try:
        return Domain(domain_str.lower())
    
    except ValueError:
        # Comprehensive domain mapping with aliases for all 16 domains
        domain_mapping = {'general'                : Domain.GENERAL,
                          'default'                : Domain.GENERAL,
                          'generic'                : Domain.GENERAL,
                          'academic'               : Domain.ACADEMIC,
                          'education'              : Domain.ACADEMIC,
                          'research'               : Domain.ACADEMIC,
                          'university'             : Domain.ACADEMIC,
                          'scholarly'              : Domain.ACADEMIC,
                          'creative'               : Domain.CREATIVE,
                          'fiction'                : Domain.CREATIVE,
                          'literature'             : Domain.CREATIVE,
                          'story'                  : Domain.CREATIVE,
                          'narrative'              : Domain.CREATIVE,
                          'ai_ml'                  : Domain.AI_ML,
                          'ai'                     : Domain.AI_ML,
                          'machinelearning'        : Domain.AI_ML,
                          'ml'                     : Domain.AI_ML,
                          'artificialintelligence' : Domain.AI_ML,
                          'neural'                 : Domain.AI_ML,
                          'software_dev'           : Domain.SOFTWARE_DEV,
                          'software'               : Domain.SOFTWARE_DEV,
                          'code'                   : Domain.SOFTWARE_DEV,
                          'programming'            : Domain.SOFTWARE_DEV,
                          'development'            : Domain.SOFTWARE_DEV,
                          'dev'                    : Domain.SOFTWARE_DEV,
                          'technical_doc'          : Domain.TECHNICAL_DOC,
                          'technical'              : Domain.TECHNICAL_DOC,
                          'tech'                   : Domain.TECHNICAL_DOC,
                          'documentation'          : Domain.TECHNICAL_DOC,
                          'docs'                   : Domain.TECHNICAL_DOC,
                          'manual'                 : Domain.TECHNICAL_DOC,
                          'engineering'            : Domain.ENGINEERING,
                          'engineer'               : Domain.ENGINEERING,
                          'technical_engineering'  : Domain.ENGINEERING,
                          'science'                : Domain.SCIENCE,
                          'scientific'             : Domain.SCIENCE,
                          'research_science'       : Domain.SCIENCE,
                          'business'               : Domain.BUSINESS,
                          'corporate'              : Domain.BUSINESS,
                          'commercial'             : Domain.BUSINESS,
                          'enterprise'             : Domain.BUSINESS,
                          'legal'                  : Domain.LEGAL,
                          'law'                    : Domain.LEGAL,
                          'contract'               : Domain.LEGAL,
                          'court'                  : Domain.LEGAL,
                          'juridical'              : Domain.LEGAL,
                          'medical'                : Domain.MEDICAL,
                          'healthcare'             : Domain.MEDICAL,
                          'clinical'               : Domain.MEDICAL,
                          'medicine'               : Domain.MEDICAL,
                          'health'                 : Domain.MEDICAL,
                          'journalism'             : Domain.JOURNALISM,
                          'news'                   : Domain.JOURNALISM,
                          'reporting'              : Domain.JOURNALISM,
                          'media'                  : Domain.JOURNALISM,
                          'press'                  : Domain.JOURNALISM,
                          'marketing'              : Domain.MARKETING,
                          'advertising'            : Domain.MARKETING,
                          'promotional'            : Domain.MARKETING,
                          'brand'                  : Domain.MARKETING,
                          'sales'                  : Domain.MARKETING,
                          'social_media'           : Domain.SOCIAL_MEDIA,
                          'social'                 : Domain.SOCIAL_MEDIA,
                          'casual'                 : Domain.SOCIAL_MEDIA,
                          'informal'               : Domain.SOCIAL_MEDIA,
                          'posts'                  : Domain.SOCIAL_MEDIA,
                          'blog_personal'          : Domain.BLOG_PERSONAL,
                          'blog'                   : Domain.BLOG_PERSONAL,
                          'personal'               : Domain.BLOG_PERSONAL,
                          'diary'                  : Domain.BLOG_PERSONAL,
                          'lifestyle'              : Domain.BLOG_PERSONAL,
                          'tutorial'               : Domain.TUTORIAL,
                          'guide'                  : Domain.TUTORIAL,
                          'howto'                  : Domain.TUTORIAL,
                          'instructional'          : Domain.TUTORIAL,
                          'educational'            : Domain.TUTORIAL,
                          'walkthrough'            : Domain.TUTORIAL,
                         }
        
        normalized_domain = domain_str.lower().strip()
        
        if normalized_domain in domain_mapping:
            return domain_mapping[normalized_domain]
        
        # Try to match with underscores/spaces variations
        normalized_with_underscores = normalized_domain.replace(' ', '_')
        if normalized_with_underscores in domain_mapping:
            return domain_mapping[normalized_with_underscores]
        
        # Try partial matching for more flexibility
        for alias, domain_enum in domain_mapping.items():
            if normalized_domain in alias or alias in normalized_domain:
                return domain_enum
        
        return None


def _validate_file_extension(filename: str) -> str:
    """
    Validate file extension and return normalized extension
    """
    file_extension     = Path(filename).suffix.lower()
    allowed_extensions = ['.txt', 
                          '.pdf', 
                          '.docx', 
                          '.doc', 
                          '.md',
                         ]
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code = 400,
                            detail      = f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
                           )
    
    return file_extension


def _generate_reasoning(detection_result: DetectionResult, attribution_result: Optional[AttributionResult] = None) -> Dict[str, Any]:
    """
    Generate detailed reasoning for detection results
    """
    if not reasoning_generator:
        return {}
    
    try:
        reasoning = reasoning_generator.generate(ensemble_result    = detection_result.ensemble_result,
                                                 metric_results     = detection_result.metric_results,
                                                 domain             = detection_result.domain_prediction.primary_domain,
                                                 attribution_result = attribution_result,
                                                 text_length        = detection_result.processed_text.word_count,
                                                )

        return safe_serialize_response(reasoning.to_dict())

    except Exception as e:
        logger.warning(f"Reasoning generation failed: {e}")
        return {}


def _generate_reports(detection_result: DetectionResult, attribution_result: Optional[AttributionResult] = None, highlighted_sentences: Optional[List] = None, 
                      analysis_id: str = None) -> Dict[str, str]:
    """
    Generate reports for detection results
    """
    if not reporter:
        return {}
    
    try:
        report_files = reporter.generate_complete_report(detection_result      = detection_result,
                                                         attribution_result    = attribution_result,
                                                         highlighted_sentences = highlighted_sentences,
                                                         formats               = ["json", "pdf"],
                                                         filename_prefix       = analysis_id or f"report_{int(time.time() * 1000)}",
                                                        )
        return report_files

    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        return {}


# ==================== ROOT & HEALTH ENDPOINTS ====================
@app.get("/", response_class = HTMLResponse)
async def root():
    """
    Serve the main web interface
    """
    # Serve the updated index.html directly from the current directory
    index_path = Path(__file__).parent / "index.html"
    
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    # Fallback to static directory if exists
    ui_static_path = Path(__file__).parent / "ui" / "static"
    index_path = ui_static_path / "index.html"
    
    if index_path.exists():
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    return HTMLResponse(content = """
                                      <html>
                                          <head><title>TEXT-AUTH API</title></head>
                                          <body style="font-family: sans-serif; padding: 50px; text-align: center;">
                                              <h1>🔍 TEXT-AUTH API</h1>
                                              <p>AI Text Detection Platform v2.0</p>
                                              <p><a href="/api/docs">API Documentation</a></p>
                                              <p><a href="/health">Health Check</a></p>
                                          </body>
                                      </html>
                                  """
                       )


@app.get("/health", response_model = HealthCheckResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthCheckResponse(status        = "healthy" if all(initialization_status.values()) else "degraded",
                               version       = "2.0.0",
                               uptime        = time.time() - app_start_time,
                               models_loaded = initialization_status,
                             )


# ==================== ANALYSIS ENDPOINTS ====================
@app.post("/api/analyze", response_model = TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text for AI generation
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time  = time.time()
    analysis_id = f"analysis_{int(time.time() * 1000)}"
    
    try:
        # Parse domain if provided
        domain = _parse_domain(request.domain)
        
        if (request.domain and not domain):
            raise HTTPException(status_code = 400,
                                detail      = f"Invalid domain. Valid options: {[d.value for d in Domain]}",
                               )
        
        # Run detection analysis
        logger.info(f"[{analysis_id}] Analyzing text ({len(request.text)} chars)")
        
        detection_result = orchestrator.analyze(text           = request.text,
                                                domain         = domain,
                                                skip_expensive = request.skip_expensive_metrics,
                                               )
        
        # Convert detection result to ensure serializability
        detection_dict   = safe_serialize_response(detection_result.to_dict())
        
        # Attribution (if enabled)
        attribution_result = None
        attribution_dict   = None
        
        if (request.enable_attribution and attributor):
            try:
                logger.info(f"[{analysis_id}] Running attribution...")
                attribution_result = attributor.attribute(text           = request.text,
                                                          processed_text = detection_result.processed_text,
                                                          metric_results = detection_result.metric_results,
                                                          domain         = detection_result.domain_prediction.primary_domain,
                                                         )

                attribution_dict   = safe_serialize_response(attribution_result.to_dict())

            except Exception as e:
                logger.warning(f"Attribution failed: {e}")
        
        # Highlighting (if enabled)
        highlighted_sentences = None
        highlighted_html      = None

        if request.enable_highlighting and highlighter:
            try:
                logger.info(f"[{analysis_id}] Generating highlights...")
                highlighted_sentences = highlighter.generate_highlights(text               = request.text,
                                                                        metric_results     = detection_result.metric_results,
                                                                        ensemble_result    = detection_result.ensemble_result,
                                                                        use_sentence_level = request.use_sentence_level,
                                                                       )

                highlighted_html      = highlighter.generate_html(highlighted_sentences = highlighted_sentences,
                                                                  include_legend        = False,
                                                                  include_metrics       = request.include_metrics_summary,
                                                                 )
            except Exception as e:
                logger.warning(f"Highlighting failed: {e}")

        
        # Generate reasoning
        reasoning_dict = _generate_reasoning(detection_result   = detection_result, 
                                             attribution_result = attribution_result,
                                            )
        
        # Generate reports (if requested)
        report_files   = dict()

        if request.generate_report:
            try:
                logger.info(f"[{analysis_id}] Generating reports...")
                report_files = _generate_reports(detection_result      = detection_result,
                                                 attribution_result    = attribution_result,
                                                 highlighted_sentences = highlighted_sentences,
                                                 analysis_id           = analysis_id,
                                                )

            except Exception as e:
                logger.warning(f"Report generation failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Cache the full analysis result
        if analysis_cache:
            cache_data = {'detection_result'      : detection_result,
                          'attribution_result'    : attribution_result,
                          'highlighted_sentences' : highlighted_sentences,
                          'original_text'         : request.text,
                          'processing_time'       : processing_time,
                         }

            analysis_cache.set(analysis_id, cache_data)

            logger.debug(f"Cached analysis: {analysis_id}")
        
        # Log the detection event
        log_detection_event(analysis_id         = analysis_id,
                            text_length         = len(request.text),
                            verdict             = detection_result.ensemble_result.final_verdict,
                            confidence          = detection_result.ensemble_result.overall_confidence,
                            domain              = detection_result.domain_prediction.primary_domain.value,
                            processing_time     = processing_time,
                            enable_attribution  = request.enable_attribution,
                            enable_highlighting = request.enable_highlighting,
                           )
        
        return TextAnalysisResponse(status           = "success",
                                    analysis_id      = analysis_id,
                                    detection_result = detection_dict,
                                    attribution      = attribution_dict,
                                    highlighted_html = highlighted_html,
                                    reasoning        = reasoning_dict,
                                    report_files     = report_files,
                                    processing_time  = processing_time,
                                    timestamp        = datetime.now().isoformat(),
                                )
        
    except HTTPException:
        central_logger.log_error("TextAnalysisError",
                                f"Analysis failed for request",
                                {"text_length": len(request.text)},
                                e,
                                )

        raise

    except Exception as e:
        logger.error(f"[{analysis_id}] Analysis failed: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.post("/api/analyze/file", response_model = FileAnalysisResponse)
async def analyze_file(file: UploadFile = File(...), domain: Optional[str] = Form(None), enable_attribution: bool = Form(True), skip_expensive_metrics: bool = Form(False),
                       use_sentence_level: bool = Form(True), include_metrics_summary: bool = Form(True), generate_report: bool = Form(False)):
    """
    Analyze uploaded document (PDF, DOCX, TXT)
    """
    if not document_extractor or not orchestrator:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )
    
    start_time  = time.time()
    analysis_id = f"file_{int(time.time() * 1000)}"

    try:
        # Validate file
        file_ext      = _validate_file_extension(file.filename)
        
        # Read and extract text
        logger.info(f"[{analysis_id}] Extracting text from {file.filename}")

        file_bytes    = await file.read()
        
        extracted_doc = document_extractor.extract_from_bytes(file_bytes = file_bytes,
                                                              filename   = file.filename,
                                                             )
        
        if not extracted_doc.is_success or not extracted_doc.text:
            raise HTTPException(status_code = 400,
                                detail      = f"Text extraction failed: {extracted_doc.error_message}"
                               )
        
        logger.info(f"[{analysis_id}] Extracted {len(extracted_doc.text)} characters")
        
        # Parse domain and analyze
        domain_enum        = _parse_domain(domain)
        
        detection_result   = orchestrator.analyze(text           = extracted_doc.text,
                                                  domain         = domain_enum,
                                                  skip_expensive = skip_expensive_metrics,
                                                 )
        
        # Set file_info on detection_result
        detection_result.file_info = {"filename"          : file.filename,
                                      "file_type"         : file_ext,
                                      "pages"             : extracted_doc.page_count,
                                      "extraction_method" : extracted_doc.extraction_method,
                                      "highlighted_html"  : False,
                                     }
        
        # Convert to serializable dict
        detection_dict     = safe_serialize_response(detection_result.to_dict())
        
        # Attribution
        attribution_result = None
        attribution_dict   = None
        
        if (enable_attribution and attributor):
            try:
                attribution_result = attributor.attribute(text           = extracted_doc.text,
                                                          processed_text = detection_result.processed_text,
                                                          metric_results = detection_result.metric_results,
                                                          domain         = detection_result.domain_prediction.primary_domain,
                                                         )

                attribution_dict   = safe_serialize_response(attribution_result.to_dict())

            except Exception as e:
                logger.warning(f"Attribution failed: {e}")
        
        # Highlighting
        highlighted_sentences = None
        highlighted_html      = None

        if highlighter:
            try:
                highlighted_sentences = highlighter.generate_highlights(text               = extracted_doc.text,
                                                                        metric_results     = detection_result.metric_results,
                                                                        ensemble_result    = detection_result.ensemble_result,
                                                                        use_sentence_level = use_sentence_level,
                                                                       )

                highlighted_html      = highlighter.generate_html(highlighted_sentences = highlighted_sentences,
                                                                  include_legend        = False,
                                                                  include_metrics       = include_metrics_summary,
                                                                 )
            except Exception as e:
                logger.warning(f"Highlighting failed: {e}")
        
        # Generate reasoning
        reasoning_dict = _generate_reasoning(detection_result   = detection_result, 
                                             attribution_result = attribution_result,
                                            )
        
        # Generate reports (if requested)
        report_files   = dict()

        if generate_report:
            try:
                logger.info(f"[{analysis_id}] Generating reports...")
                report_files = _generate_reports(detection_result      = detection_result,
                                                 attribution_result    = attribution_result,
                                                 highlighted_sentences = highlighted_sentences,
                                                 analysis_id           = analysis_id,
                                                )
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Cache the full analysis result including Original Text 
        if analysis_cache:
            cache_data = {'detection_result'      : detection_result,
                          'attribution_result'    : attribution_result,
                          'highlighted_sentences' : highlighted_sentences,
                          'original_text'         : extracted_doc.text, 
                          'processing_time'       : processing_time,
                         }

            analysis_cache.set(analysis_id, cache_data)
            logger.info(f"✓ Cached file analysis: {analysis_id} (text_length={len(extracted_doc.text)})")
        
        return FileAnalysisResponse(status           = "success",
                                    analysis_id      = analysis_id,
                                    file_info        = {"filename"          : file.filename,
                                                        "file_type"         : file_ext,
                                                        "pages"             : extracted_doc.page_count,
                                                        "extraction_method" : extracted_doc.extraction_method,
                                                        "highlighted_html"  : highlighted_html is not None,
                                                       },
                                    detection_result = detection_dict,
                                    attribution      = attribution_dict,
                                    highlighted_html = highlighted_html,
                                    reasoning        = reasoning_dict,
                                    report_files     = report_files,
                                    processing_time  = processing_time,
                                    timestamp        = datetime.now().isoformat(),
                                   )
        
    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"[{analysis_id}] File analysis failed: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


@app.post("/api/analyze/batch", response_model = BatchAnalysisResponse)
async def batch_analyze(request: BatchAnalysisRequest):
    """
    Analyze multiple texts in batch
    - Limits : 1-100 texts per request
    """
    if not orchestrator:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )

    if (len(request.texts) > 100):
        raise HTTPException(status_code = 400, 
                            detail      = "Maximum 100 texts per batch",
                           )


    start_time = time.time()
    batch_id   = f"batch_{int(time.time() * 1000)}"

    try:
        # Parse domain
        domain  = _parse_domain(request.domain)
        
        logger.info(f"[{batch_id}] Processing {len(request.texts)} texts")
        
        results = list()

        for i, text in enumerate(request.texts):
            try:
                detection_result   = orchestrator.analyze(text           = text,
                                                          domain         = domain,
                                                          skip_expensive = request.skip_expensive_metrics,
                                                         )
                
                # Convert to serializable dict
                detection_dict     = safe_serialize_response(detection_result.to_dict())
                
                # Attribution if enabled
                attribution_result = None
                attribution_dict   = None

                if request.enable_attribution and attributor:
                    try:
                        attribution_result = attributor.attribute(text           = text,
                                                                  processed_text = detection_result.processed_text,
                                                                  metric_results = detection_result.metric_results,
                                                                  domain         = detection_result.domain_prediction.primary_domain,
                                                                 )

                        attribution_dict   = safe_serialize_response(attribution_result.to_dict())

                    except Exception:
                        pass
                
                # Generate reasoning
                reasoning_dict = _generate_reasoning(detection_result   = detection_result, 
                                                     attribution_result = attribution_result,
                                                    )
                
                # Generate reports if requested
                report_files   = dict()

                if request.generate_reports:
                    try:
                        report_files = _generate_reports(detection_result   = detection_result,
                                                         attribution_result = attribution_result,
                                                         analysis_id        = f"{batch_id}_{i}"
                                                        )
                    except Exception:
                        pass
                
                results.append(BatchAnalysisResult(index        = i,
                                                   status       = "success",
                                                   detection    = detection_dict,
                                                   attribution  = attribution_dict,
                                                   reasoning    = reasoning_dict,
                                                   report_files = report_files,
                                                  )
                              )
                
            except Exception as e:
                logger.error(f"[{batch_id}] Text {i} failed: {e}")
                results.append(BatchAnalysisResult(index  = i,
                                                   status = "error",
                                                   error  = str(e),
                                                  )
                              )
        
        processing_time = time.time() - start_time
        success_count   = sum(1 for r in results if r.status == "success")
        
        logger.success(f"[{batch_id}] Batch complete: {success_count}/{len(request.texts)} successful")
        
        return BatchAnalysisResponse(status          = "success",
                                     batch_id        = batch_id,
                                     total           = len(request.texts),
                                     successful      = success_count,
                                     failed          = len(request.texts) - success_count,
                                     results         = results,
                                     processing_time = processing_time,
                                     timestamp       = datetime.now().isoformat(),
                                    )
        
    except Exception as e:
        logger.error(f"[{batch_id}] Batch analysis failed: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )


# ==================== REPORT GENERATION ENDPOINTS ====================
@app.post("/api/report/generate", response_model = ReportGenerationResponse)
async def generate_report(background_tasks: BackgroundTasks, analysis_id: str = Form(...), formats: str = Form("json,pdf"), include_highlights: bool = Form(True)):
    """
    Generate detailed report for a cached analysis
    """
    if not orchestrator or not reporter or not analysis_cache:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )

    try:
        # Check cache first
        cached_data = analysis_cache.get(analysis_id)
        
        if not cached_data:
            raise HTTPException(status_code = 404, 
                                detail      = f"Analysis {analysis_id} not found in cache. Please run the analysis first, then request the report.",
                               )
        
        logger.info(f"Using cached analysis for report generation: {analysis_id}")

        # Extract cached data
        detection_result      = cached_data['detection_result']
        attribution_result    = cached_data.get('attribution_result')
        highlighted_sentences = cached_data.get('highlighted_sentences')
        
        # Parse formats
        requested_formats     = [f.strip() for f in formats.split(',')]
        valid_formats         = ['json', 'pdf']
        
        for fmt in requested_formats:
            if fmt not in valid_formats:
                raise HTTPException(status_code = 400,
                                    detail      = f"Invalid format '{fmt}'. Valid: {', '.join(valid_formats)}",
                                   )
        
        # Generate reports using cached data
        logger.info(f"Generating {', '.join(requested_formats)} report(s) for {analysis_id}")
        
        report_files     = reporter.generate_complete_report(detection_result      = detection_result,
                                                             attribution_result    = attribution_result,
                                                             highlighted_sentences = highlighted_sentences if include_highlights else None,
                                                             formats               = requested_formats,
                                                             filename_prefix       = analysis_id,
                                                            )

        # Extract only the filename from the full path for the response
        report_filenames = dict()

        for fmt, full_path in report_files.items():
            report_filenames[fmt] = Path(full_path).name
        
        logger.success(f"Generated {len(report_filenames)} report(s) for {analysis_id}")
        
        return ReportGenerationResponse(status      = "success",
                                        analysis_id = analysis_id,
                                        reports     = report_filenames,
                                        timestamp   = datetime.now().isoformat(),
                                       )
        
    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code = 500, 
                            detail      = str(e),
                           )

@app.get("/api/report/download/{filename}")
async def download_report(filename: str):
    """
    Download a generated report
    """
    if not reporter:
        raise HTTPException(status_code = 503,
                            detail      = "Service not initialized",
                           )

    file_path = reporter.output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code = 404, 
                            detail      = "Report not found",
                           )

    return FileResponse(path       = str(file_path),
                        filename   = filename,
                        media_type = "application/octet-stream",
                       )


# ==================== UTILITY ENDPOINTS ====================
@app.get("/api/domains")
async def list_domains():
    """
    List all supported domains
    """
    domains_list = list()
    for domain in Domain:
        domains_list.append({"value"       : domain.value,
                             "name"        : domain.value.replace('_', ' ').title(),
                             "description" : _get_domain_description(domain),
                           })

    return {"domains": domains_list}


@app.get("/api/models")
async def list_ai_models():
    """
    List all AI models that can be attributed
    """
    return {"models" : [{"value" : model.value,
                         "name"  : model.value.replace('-', ' ').replace('_', ' ').title(),
                        }
                        for model in AIModel if model not in [AIModel.HUMAN, AIModel.UNKNOWN]
                       ]
           }


@app.get("/api/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics (admin endpoint)
    """
    if not analysis_cache:
        return {"status" : "cache not initialized"}

    return {"cache_size"  : analysis_cache.size(),
            "max_size"    : analysis_cache.max_size,
            "ttl_seconds" : analysis_cache.ttl_seconds,
           }


@app.post("/api/cache/clear")
async def clear_cache():
    """
    Clear analysis cache (admin endpoint)
    """
    if not analysis_cache:
        raise HTTPException(status_code = 503, 
                            detail      = "Cache not initialized",
                           )

    analysis_cache.clear()

    return {"status"  : "success", 
            "message" : "Cache cleared",
           }


# ==================== ERROR HANDLERS ====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Handle HTTP exceptions
    """
    return NumpyJSONResponse(status_code = exc.status_code,
                             content     = ErrorResponse(status    = "error",
                                                         error     = exc.detail,
                                                         timestamp = datetime.now().isoformat(),
                                                        ).dict()
                            )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Handle general exceptions
    """
    logger.error(f"Unhandled exception: {exc}")
    return NumpyJSONResponse(status_code = 500,
                             content     = ErrorResponse(status    = "error",
                                                         error     = "Internal server error",
                                                         timestamp = datetime.now().isoformat(),
                                                        ).dict()
                            )


# Add middleware for API request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time   = time.time()
    response     = await call_next(request)
    process_time = time.time() - start_time

    log_api_request(method      = request.method,
                    path        = request.url.path,
                    status_code = response.status_code,
                    duration    = process_time,
                    ip          = request.client.host if request.client else None,
                   )

    return response



# ==================== MAIN ====================
if __name__ == "__main__":
    # Configure logging
    log_level = settings.LOG_LEVEL.lower()
    logger.info("Starting TEXT-AUTH API Server...")

    uvicorn.run("text_auth_app:app",
                host       = settings.HOST,
                port       = settings.PORT,
                reload     = settings.DEBUG,
                log_level  = log_level,
                workers    = 1 if settings.DEBUG else settings.WORKERS,
               )
