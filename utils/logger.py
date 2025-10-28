# DEPENDENCIES
import os
import sys
import json
import time
import logging
from typing import Any
from typing import Dict
from pathlib import Path
from loguru import logger
from typing import Optional
from datetime import datetime
from config.settings import settings


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages toward Loguru
    """
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to Loguru
        """
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while (frame.f_code.co_filename == logging.__file__):
            frame  = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class JSONFormatter:
    """
    JSON formatter for structured logging
    """
    def __init__(self):
        self.pid = os.getpid()
    

    def format(self, record: Dict[str, Any]) -> str:
        """
        Format log record as JSON
        """
        # Create structured log entry
        log_entry = {"timestamp"  : datetime.fromtimestamp(record["time"].timestamp()).isoformat(),
                     "level"      : record["level"].name,
                     "message"    : record["message"],
                     "module"     : record["name"],
                     "function"   : record["function"],
                     "line"       : record["line"],
                     "process_id" : self.pid,
                     "thread_id"  : record["thread"].id if record.get("thread") else None,
                    }
        
        # Add exception info if present
        if record.get("exception"):
            log_entry["exception"] = {"type"      : str(record["exception"].type),
                                      "value"     : str(record["exception"].value),
                                      "traceback" : "".join(record["exception"].traceback).strip() if record["exception"].traceback else None,
                                     }
        
        # Add extra fields
        if record.get("extra"):
            log_entry.update(record["extra"])
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class CentralizedLogger:
    """
    Centralized logging system for AI Text Detector
    
    Features:
    - Structured JSON logging for production
    - Human-readable console logging for development
    - Automatic log rotation and retention
    - Integration with standard logging and Loguru
    - Performance monitoring
    - Security event logging
    """
    
    def __init__(self):
        self.initialized = False
        self.log_dir     = Path(__file__).parent.parent / "logs"

        self.setup_log_dir()

    
    def setup_log_dir(self) -> None:
        """
        Create log directory structure
        """
        try:
            self.log_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (self.log_dir / "application").mkdir(exist_ok = True)
            (self.log_dir / "performance").mkdir(exist_ok = True)
            (self.log_dir / "security").mkdir(exist_ok = True)
            (self.log_dir / "errors").mkdir(exist_ok = True)
            
            logger.info(f"Log directory structure created at: {self.log_dir}")
        
        except Exception as e:
            print(f"CRITICAL: Failed to create log directory: {e}")
            sys.exit(1)
    

    def initialize(self) -> bool:
        """
        Initialize centralized logging system
        
        Returns:
        --------
            { bool } : True if successful, False otherwise
        """
        try:
            # Remove default logger
            logger.remove()
            
            # Configure based on environment
            if (settings.ENVIRONMENT == "production"):
                self._setup_production_logging()
            
            else:
                self._setup_development_logging()
            
            # Intercept standard logging
            self._intercept_standard_logging()
            
            # Log initialization
            logger.success("Centralized logging system initialized")
            logger.info(f"Environment: {settings.ENVIRONMENT}")
            logger.info(f"Log Level: {settings.LOG_LEVEL}")
            logger.info(f"Log Directory: {self.log_dir}")
            
            self.initialized = True
            return True
        
        except Exception as e:
            print(f"CRITICAL: Failed to initialize logging: {e}")
            return False
    

    def _setup_production_logging(self) -> None:
        """
        Setup production logging with JSON format and rotation
        """
        # Application logs (all events)
        logger.add(self.log_dir / "application" / "app_{time:YYYY-MM-DD}.log",
                   format      = "{message}",
                   filter      = lambda record: record["extra"].get("log_type", "application") == "application",
                   level       = settings.LOG_LEVEL,
                   rotation    = "00:00",    # Rotate daily at midnight
                   retention   = "30 days",  # Keep logs for 30 days
                   compression = "gz",       # Compress old logs
                   serialize   = True,       # Output as JSON
                   backtrace   = True,
                   diagnose    = True,
                   enqueue     = True,       # Thread-safe
                  )
        
        # Performance logs
        logger.add(self.log_dir / "performance" / "performance_{time:YYYY-MM-DD}.log",
                   format      = "{message}",
                   filter      = lambda record: record["extra"].get("log_type") == "performance",
                   level       = "INFO",
                   rotation    = "00:00",
                   retention   = "7 days",
                   compression = "gz",
                   serialize   = True,
                   backtrace   = False,
                   diagnose    = False,
                   enqueue     = True,
                  )
        
        # Security logs
        logger.add(self.log_dir / "security" / "security_{time:YYYY-MM-DD}.log",
                   format      = "{message}",
                   filter      = lambda record: record["extra"].get("log_type") == "security",
                   level       = "INFO",
                   rotation    = "00:00",
                   retention   = "90 days",  # Keep security logs longer
                   compression = "gz",
                   serialize   = True,
                   backtrace   = True,
                   diagnose    = True,
                   enqueue     = True,
                  )
        
        # Error logs (separate file for easier monitoring)
        logger.add(self.log_dir / "errors" / "errors_{time:YYYY-MM-DD}.log",
                   format      = "{message}",
                   filter      = lambda record: record["level"].name in ["ERROR", "CRITICAL"],
                   level       = "ERROR",
                   rotation    = "00:00",
                   retention   = "30 days",
                   compression = "gz",
                   serialize   = True,
                   backtrace   = True,
                   diagnose    = True,
                   enqueue     = True,
                  )
        
        # Console output for production (JSON format)
        logger.add(sys.stderr,
                   format    = "{message}",
                   level     = settings.LOG_LEVEL,
                   serialize = True,
                   backtrace = True,
                   diagnose  = settings.DEBUG,
                  )

    
    def _setup_development_logging(self) -> None:
        """
        Setup development logging with human-readable format
        """
        # Colorful console output for development
        logger.add(sys.stderr, 
                   format    = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                               "<level>{level: <8}</level> | "
                               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                               "<level>{message}</level>",
                   level     = settings.LOG_LEVEL,
                   colorize  = True,
                   backtrace = True,
                   diagnose  = True,
                   enqueue   = True,
                )
        
        # File logging for development (structured)
        logger.add(self.log_dir / "application" / "app_{time:YYYY-MM-DD}.log",
                   format      = "{message}",
                   level       = settings.LOG_LEVEL,
                   rotation    = "10 MB",  # Rotate by size in development
                   retention   = "7 days",
                   compression = "gz",
                   serialize   = True,
                   backtrace   = True,
                   diagnose    = True,
                   enqueue     = True,
                  )
    

    def _intercept_standard_logging(self) -> None:
        """
        Intercept standard library logging
        """
        # Get root logger
        logging.root.setLevel(settings.LOG_LEVEL.upper())
        
        # Remove existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Add intercept handler
        intercept_handler = InterceptHandler()
        logging.root.addHandler(intercept_handler)
        
        # Intercept third-party loggers
        for log_name in logging.root.manager.loggerDict.keys():
            if log_name.startswith(("uvicorn", "fastapi", "detector", "processor")):
                logging.getLogger(log_name).handlers  = [intercept_handler]
                logging.getLogger(log_name).propagate = False
    

    def get_logger(self, name: Optional[str] = None):
        """
        Get a logger instance with context
        
        Arguments:
        ----------
            name { str } : Logger name (usually __name__)
            
        Returns:
        --------
            Logger instance
        """
        if name:
            return logger.bind(logger_name = name)

        return logger
    

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """
        Log performance metrics
        
        Arguments:
        ----------
            operation { str }   : Operation name 

            duration  { float } : Duration in seconds
            
            **kwargs            : Additional performance metrics
        """
        performance_data = {"operation"        : operation,
                            "duration_seconds" : round(duration, 4),
                            "timestamp"        : datetime.now().isoformat(),
                            **kwargs
                           }
        
        logger.bind(log_type = "performance").info(f"Performance metric: {operation}",
                    extra    = performance_data,
                   )

    
    def log_security_event(self, event_type: str, user: Optional[str] = None, ip: Optional[str] = None, **kwargs) -> None:
        """
        Log security events
        
        Arguments:
        ----------
            event_type { str } : Type of security event
            
            user       { str } : User identifier (if available)
            
            ip         { str } : IP address (if available)
            
            **kwargs           : Additional security context
        """
        security_data = {"event_type" : event_type,
                         "user"       : user,
                         "ip_address" : ip,
                         "timestamp"  : datetime.now().isoformat(),
                         **kwargs,
                        }
        
        logger.bind(log_type = "security").warning(f"Security event: {event_type}",
                    extra    = security_data,
                   )
    
    def log_api_request(self, method: str, path: str, status_code: int, duration: float, user: Optional[str] = None, ip: Optional[str] = None, **kwargs) -> None:
        """
        Log API request details
        
        Arguments:
        ----------
            method      { str }   : HTTP method

            path        { str }   : Request path
            
            status_code { int }   : HTTP status code
            
            duration    { float } : Request duration in seconds
            
            user        { str }   : User identifier
            
            ip          { str }   : Client IP address
            
            **kwargs              : Additional request context
        """
        request_data = {"http_method"      : method,
                        "path"             : path,
                        "status_code"      : status_code,
                        "duration_seconds" : round(duration, 4),
                        "user"             : user,
                        "ip_address"       : ip,
                        "timestamp"        : datetime.now().isoformat(),
                        **kwargs
                       }
        
        # Log as info for successful requests, warning for client errors, error for server errors
        if (status_code < 400):
            logger.bind(log_type = "application").info(f"API Request: {method} {path} -> {status_code}",
                        extra    = request_data,
                       )

        elif (status_code < 500):
            logger.bind(log_type = "application").warning(f"API Client Error: {method} {path} -> {status_code}",
                        extra    = request_data,
                       )

        else:
            logger.bind(log_type = "application").error(f"API Server Error: {method} {path} -> {status_code}",
                        extra    = request_data,
                       )
    

    def log_detection_event(self, analysis_id: str, text_length: int, verdict: str, confidence: float, domain: str, processing_time: float, **kwargs) -> None:
        """
        Log text detection events
        
        Arguments:
        ----------
            analysis_id     { str }   : Unique analysis identifier

            text_length     { int }   : Length of analyzed text
            
            verdict         { str }   : Detection verdict
            
            confidence      { float } : Confidence score
            
            domain          { str }   : Content domain
            
            processing_time { float } : Processing time in seconds
            
            **kwargs                  : Additional detection context
        """
        detection_data = {"analysis_id"             : analysis_id,
                          "text_length"             : text_length,
                          "verdict"                 : verdict,
                          "confidence"              : round(confidence, 4),
                          "domain"                  : domain,
                          "processing_time_seconds" : round(processing_time, 4),
                          "timestamp"               : datetime.now().isoformat(),
                          **kwargs
                        }
        
        logger.bind(log_type = "application").info(f"Detection completed: {analysis_id} -> {verdict}",
                    extra    = detection_data,
                   )

    
    def log_model_loading(self, model_name: str, success: bool, load_time: float, **kwargs) -> None:
        """
        Log model loading events
        
        Arguments:
        ----------
            model_name  { str }  : Name of the model

            success     { bool } : Whether loading was successful
            
            load_time  { float } : Loading time in seconds
            
            **kwargs             : Additional model context
        """
        model_data = {"model_name"        : model_name,
                      "success"           : success,
                      "load_time_seconds" : round(load_time, 4),
                      "timestamp"         : datetime.now().isoformat(),
                      **kwargs
                     }
        
        if success:
            logger.bind(log_type = "application").info(f"Model loaded: {model_name}",
                        extra    = model_data,
                       )

        else:
            logger.bind(log_type = "application").error(f"Model failed to load: {model_name}",
                        extra    = model_data,
                       )

    
    def log_error(self, error_type: str, message: str, context: Dict[str, Any] = None, exception: Optional[Exception] = None) -> None:
        """
        Log error with context
        
        Arguments:
        ----------
            error_type { str }       : Type of error

            message    { str }       : Error message
            
            context    { dict }      : Error context
            
            exception  { Exception } : Exception object
        """
        error_data = {"error_type" : error_type,
                      "message"    : message,
                      "context"    : context or {},
                      "timestamp"  : datetime.now().isoformat(),
                     }
        
        if exception:
            error_data["exception"] = {"type"    : type(exception).__name__,
                                       "message" : str(exception),
                                      }
        
        logger.bind(log_type  = "application").error(f"Error: {error_type} - {message}",
                    extra     = error_data,
                    exception = exception,
                   )
    

    def log_startup(self, component: str, success: bool, **kwargs) -> None:
        """
        Log application startup events
        
        Arguments:
        ----------
            component { str }  : Component name

            success   { bool } : Whether startup was successful

            **kwargs           : Additional startup context
        """
        startup_data = {"component" : component,
                        "success"   : success,
                        "timestamp" : datetime.now().isoformat(),
                        **kwargs
                       }
        
        if success:
            logger.bind(log_type = "application").info(f"Startup: {component} initialized",
                        extra    = startup_data,
                       )

        else:
            logger.bind(log_type = "application").error(f"Startup: {component} failed",
                        extra    = startup_data,
                       )

    
    def cleanup(self) -> None:
        """
        Cleanup logging resources
        """
        try:
            logger.complete()
            logger.info("Logging system cleanup completed")
        
        except Exception as e:
            print(f"Error during logging cleanup: {e}")


# Global logger instance
central_logger = CentralizedLogger()


# Convenience functions for direct usage
def get_logger(name: Optional[str] = None):
    """
    Get a logger instance
    
    Arguments:
    ----------
        name { str } : Logger name
        
    Returns:
    --------
        Logger instance
    """
    return central_logger.get_logger(name)


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics
    """
    central_logger.log_performance(operation, duration, **kwargs)


def log_security_event(event_type: str, user: Optional[str] = None, ip: Optional[str] = None, **kwargs) -> None:
    """
    Log security events
    """
    central_logger.log_security_event(event_type, user, ip, **kwargs)


def log_api_request(method: str, path: str, status_code: int, duration: float, user: Optional[str] = None, ip: Optional[str] = None, **kwargs) -> None:
    """
    Log API request details
    """
    central_logger.log_api_request(method, path, status_code, duration, user, ip, **kwargs)


def log_detection_event(analysis_id: str, text_length: int,  verdict: str, confidence: float, domain: str, processing_time: float, **kwargs) -> None:
    """
    Log text detection events
    """
    central_logger.log_detection_event(analysis_id, text_length, verdict, confidence, domain, processing_time, **kwargs)


def log_model_loading(model_name: str, success: bool, load_time: float, **kwargs) -> None:
    """
    Log model loading events
    """
    central_logger.log_model_loading(model_name, success, load_time, **kwargs)


def log_error(error_type: str, message: str, context: Dict[str, Any] = None, exception: Optional[Exception] = None) -> None:
    """
    Log error with context
    """
    central_logger.log_error(error_type, message, context, exception)


def log_startup(component: str, success: bool, **kwargs) -> None:
    """
    Log application startup events
    """
    central_logger.log_startup(component, success, **kwargs)




# Export
__all__ = ["log_error",
           "get_logger",
           "log_startup",
           "central_logger",
           "log_performance",
           "log_api_request",
           "CentralizedLogger",
           "log_model_loading",
           "log_security_event",
           "log_detection_event",
          ]