# DEPENDENCIES
import os
import torch
from pathlib import Path
from pydantic import Field
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Main application settings
    """
    # Application Info
    APP_NAME                : str           = "TEXT-AUTH"
    APP_VERSION             : str           = "1.0.0"
    APP_DESCRIPTION         : str           = "AI Text Detection Platform"
    
    # Environment
    ENVIRONMENT             : str           = Field(default = "development", env = "ENVIRONMENT")
    DEBUG                   : bool          = Field(default = True, env = "DEBUG")
    
    # Server Configuration
    HOST                    : str           = Field(default = "0.0.0.0", env = "HOST")
    PORT                    : int           = Field(default = 8000, env = "PORT")
    WORKERS                 : int           = Field(default = 4, env = "WORKERS")
    
    # Paths
    BASE_DIR                : Path          = Path(__file__).parent.parent.resolve()
    MODEL_CACHE_DIR         : Path          = Field(default = Path(__file__).parent.parent / "models" / "cache", env = "MODEL_CACHE_DIR")
    LOG_DIR                 : Path          = Field(default = Path(__file__).parent.parent / "logs", env = "LOG_DIR")
    UPLOAD_DIR              : Path          = Field(default = Path(__file__).parent.parent / "data" / "uploads", env = "UPLOAD_DIR")
    REPORT_DIR              : Path          = Field(default = Path(__file__).parent.parent / "data" / "reports", env = "REPORT_DIR")
    
    # File Upload Settings
    MAX_UPLOAD_SIZE         : int           = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS      : list          = [".txt", ".pdf", ".docx", ".doc", ".md"]
    
    # Processing Settings
    MAX_TEXT_LENGTH         : int           = 500000  # Maximum characters to process
    MIN_TEXT_LENGTH         : int           = 50      # Minimum characters for analysis
    CHUNK_SIZE              : int           = 512     # Tokens per chunk
    CHUNK_OVERLAP           : int           = 50      # Overlap between chunks
    
    # Model Settings
    DEVICE                  : str           = Field(default = "cpu", env = "DEVICE")  # "cuda" or "cpu"
    USE_QUANTIZATION        : bool          = Field(default = False, env = "USE_QUANTIZATION")
    USE_ONNX                : bool          = Field(default = False, env = "USE_ONNX")
    MODEL_LOAD_STRATEGY     : str           = "lazy"  # "lazy" or "eager"
    MAX_CACHED_MODELS       : int           = 5
    
    # Detection Settings
    CONFIDENCE_THRESHOLD    : float         = 0.7  # Minimum confidence for classification
    ENSEMBLE_METHOD         : str           = "weighted_average"  # "weighted_average", "voting", "stacking"
    USE_DOMAIN_ADAPTATION   : bool          = True
    
    # Rate Limiting
    RATE_LIMIT_ENABLED      : bool          = True
    RATE_LIMIT_REQUESTS     : int           = 100
    RATE_LIMIT_WINDOW       : int           = 3600  # seconds (1 hour)
    
    # Logging
    LOG_LEVEL               : str           = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT              : str           = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_ROTATION            : str           = "1 day"
    LOG_RETENTION           : str           = "30 days"
    
    # API Settings
    API_PREFIX              : str           = "/api/v1"
    CORS_ORIGINS            : list          = ["*"]  # For production, specify exact origins
    
    # Database (Optional - for future)
    DATABASE_URL            : Optional[str] = Field(default = None, env = "DATABASE_URL")
    
    # Security
    SECRET_KEY              : str           = Field(default = "your-secret-key-change-in-production", env = "SECRET_KEY")
    API_KEY_ENABLED         : bool          = False
    
    # Feature Flags
    ENABLE_ATTRIBUTION      : bool          = True
    ENABLE_HIGHLIGHTING     : bool          = True
    ENABLE_PDF_REPORTS      : bool          = True
    ENABLE_BATCH_PROCESSING : bool          = True
     
    # Performance
    MAX_CONCURRENT_REQUESTS : int           = 10
    REQUEST_TIMEOUT         : int           = 300  # seconds (5 minutes)
    
    # Metrics Configuration
    METRICS_ENABLED         : dict          = {"semantic_analysis"            : True,
                                               "multi_perturbation_stability" : True,
                                               "perplexity"                   : True,
                                               "statistical"                  : True,
                                               "entropy"                      : True,
                                               "linguistic"                   : True,
                                              }
    
    class Config:
        env_file       = ".env"
        case_sensitive = True
        extra          = "ignore"

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    

    def _create_directories(self):
        """
        Create necessary directories if they don't exist
        """
        for directory in [self.MODEL_CACHE_DIR, self.LOG_DIR, self.UPLOAD_DIR, self.REPORT_DIR]:
            directory.mkdir(parents = True, exist_ok = True)
    

    @property
    def is_production(self) -> bool:
        """
        Check if running in production
        """
        return self.ENVIRONMENT.lower() == "production"
    

    @property
    def use_gpu(self) -> bool:
        """
        Check if GPU is available and should be used
        """
        return self.DEVICE == "cuda" and torch.cuda.is_available()



# Singleton instance
settings = Settings()


# Export for easy import
__all__  = ["settings", 
            "Settings",
           ]