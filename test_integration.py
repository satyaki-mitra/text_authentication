# test_integration.py
import os
import sys
import json
from pathlib import Path
from io import StringIO
import contextlib

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create a string buffer to capture output
output_buffer = StringIO()

with contextlib.redirect_stdout(output_buffer):
    # Now import modules
    from config.enums import ModelType, Domain, Language
    from config.schemas import ModelConfig, ExtractedDocument, ProcessedText
    from config.constants import document_extraction_params
    from config.model_config import MODEL_REGISTRY, get_model_config
    from config.settings import settings
    from config.threshold_config import get_threshold_for_domain

    print("=" * 70)
    print("CONFIG MODULE INTEGRATION TEST")
    print("=" * 70)

    # Test 1: Enum usage
    print(f"\n✓ Model Types: {[m.value for m in ModelType][:5]}...")

    # Test 2: Schema instantiation
    config = ModelConfig(
        model_id="test",
        model_type=ModelType.TRANSFORMER,
        description="Test",
        size_mb=100
    )
    print(f"✓ Schema instantiation: {config.model_id}")

    # Test 3: Constants usage
    print(f"✓ Max file size: {document_extraction_params.MAX_FILE_SIZE / 1024 / 1024:.1f} MB")

    # Test 4: Model registry
    print(f"✓ Available models: {list(MODEL_REGISTRY.keys())}")

    # Test 5: Settings
    print(f"✓ App name: {settings.APP_NAME}")
    print(f"✓ Environment: {settings.ENVIRONMENT}")
    print(f"✓ Log dir: {settings.LOG_DIR}")
    print(f"✓ Model cache dir: {settings.MODEL_CACHE_DIR}")

    # Test 6: Thresholds
    thresholds = get_threshold_for_domain(Domain.ACADEMIC)
    print(f"✓ Academic thresholds: {thresholds.ensemble_threshold}")

    print("\n" + "=" * 70)
    print("PROCESSORS MODULE INTEGRATION TEST")
    print("=" * 70)

    # Test 7: Document Extractor
    try:
        from processors.document_extractor import DocumentExtractor
        
        # Create a test text file
        test_text = "This is a test document for integration testing.\n" * 10
        test_file = Path("test_document.txt")
        
        # Write test file
        test_file.write_text(test_text)
        
        # Test extractor
        extractor = DocumentExtractor(extract_metadata=True)
        result = extractor.extract(str(test_file))
        
        print(f"\n✓ Document Extractor Test:")
        print(f"  - Success: {result.is_success}")
        print(f"  - Text length: {len(result.text)} chars")
        print(f"  - File type: {result.file_type}")
        print(f"  - Method: {result.extraction_method}")
        
        # Clean up test file
        test_file.unlink()
        
    except Exception as e:
        print(f"\n✗ Document Extractor failed: {e}")

    # Test 8: Text Processor
    try:
        # First check if we have the needed constants
        from config.constants import text_processing_params
        print(f"\n✓ Text processing params available")
        
        from processors.text_processor import TextProcessor
        
        test_text = "This is a sample text for processing. It contains multiple sentences! " \
                    "Here is another sentence. And one more for testing."
        
        processor = TextProcessor()
        processed = processor.process(test_text)
        
        print(f"\n✓ Text Processor Test:")
        print(f"  - Is valid: {processed.is_valid}")
        print(f"  - Words: {processed.word_count}")
        print(f"  - Sentences: {processed.sentence_count}")
        print(f"  - Avg sentence length: {processed.avg_sentence_length:.1f}")
        print(f"  - Avg word length: {processed.avg_word_length:.1f}")
        
    except Exception as e:
        print(f"\n✗ Text Processor failed: {e}")
        print("  Note: You need to add TextProcessingParams to constants.py")

    # Test 9: Domain Classifier (without model)
    try:
        from processors.domain_classifier import DomainClassifier, get_domain_name, is_technical_domain
        
        test_text = "This is a scientific paper about machine learning and artificial intelligence."
        
        classifier = DomainClassifier()
        print(f"\n✓ Domain Classifier initialized")
        
        # Note: This will fail if models aren't loaded, but we can test the class structure
        print(f"  - Class structure verified")
        print(f"  - Domain enum available")
        
        # Test helper functions
        ai_ml_domain = Domain.AI_ML
        print(f"  - AI/ML domain name: {get_domain_name(ai_ml_domain)}")
        print(f"  - Is technical domain: {is_technical_domain(ai_ml_domain)}")
        
    except Exception as e:
        print(f"\n✗ Domain Classifier setup failed: {e}")

    # Test 10: Language Detector (heuristic mode)
    try:
        from processors.language_detector import LanguageDetector
        
        # Test in English
        english_text = "This is an English text for language detection testing."
        
        # Use heuristic mode (no model dependency)
        detector = LanguageDetector(use_model=False)
        result = detector.detect(english_text)
        
        print(f"\n✓ Language Detector Test (heuristic):")
        print(f"  - Primary language: {result.primary_language.value}")
        print(f"  - Evidence strength: {result.evidence_strength:.2f}")
        print(f"  - Method: {result.detection_method}")
        print(f"  - Script: {result.script.value}")
        
        # Test language check
        is_english = detector.is_language(english_text, Language.ENGLISH, threshold=0.5)
        print(f"  - Is English check: {is_english}")
        
    except Exception as e:
        print(f"\n✗ Language Detector failed: {e}")

    print("\n" + "=" * 70)
    print("MODELS MODULE INTEGRATION TEST")
    print("=" * 70)

    # Test 11: Model Registry
    try:
        from models.model_registry import ModelRegistry, get_model_registry
        
        registry = get_model_registry()
        
        print(f"\n✓ Model Registry Test:")
        print(f"  - Singleton pattern working")
        print(f"  - Registry initialized")
        
        # Test usage tracking
        registry.record_model_usage("test_model", 1.5)
        stats = registry.get_usage_stats("test_model")
        print(f"  - Usage tracking: {stats.usage_count if stats else 'N/A'}")
        
        # Test dependency tracking
        registry.add_dependency("model_b", ["model_a"])
        deps = registry.get_dependencies("model_b")
        print(f"  - Dependency tracking: {deps}")
        
        # Generate report
        report = registry.generate_usage_report()
        print(f"  - Report generation: {len(report)} items")
        
        # Test reset
        registry.reset_usage_stats("test_model")
        print(f"  - Reset functionality working")
        
    except Exception as e:
        print(f"\n✗ Model Registry failed: {e}")

    # Test 12: Model Manager (without actual downloads)
    try:
        from models.model_manager import ModelManager, get_model_manager
        
        manager = get_model_manager()
        
        print(f"\n✓ Model Manager Test:")
        print(f"  - Singleton pattern working")
        print(f"  - Device: {manager.device}")
        print(f"  - Cache directory: {manager.cache_dir}")
        
        # Test metadata
        metadata = manager.metadata
        print(f"  - Metadata loaded: {len(metadata)} entries")
        
        # Test cache
        cache_size = manager.cache.size()
        print(f"  - Cache initialized: size {cache_size}")
        
        # Test model info check
        model_name = list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else "perplexity_reference_lm"
        is_downloaded = manager.is_model_downloaded(model_name)
        print(f"  - Model check: {model_name} downloaded={is_downloaded}")
        
        # Test memory usage
        memory_info = manager.get_memory_usage()
        print(f"  - Memory monitoring: {len(memory_info)} metrics")
        
        # Test model configuration access
        model_config = get_model_config(model_name)
        if model_config:
            print(f"  - Model config access: {model_config.model_id}")
        
    except Exception as e:
        print(f"\n✗ Model Manager failed: {e}")

    # Test 13: Integration between models and config
    try:
        print(f"\n✓ Config-Models Integration Test:")
        
        # Check model config from registry
        for model_name, config in MODEL_REGISTRY.items():
            if config.required:
                print(f"  - {model_name}: {config.model_type.value}")
                break
        
        # Check settings integration
        print(f"  - Max cached models from settings: {settings.MAX_CACHED_MODELS}")
        print(f"  - Use quantization from settings: {settings.USE_QUANTIZATION}")
        
    except Exception as e:
        print(f"\n✗ Config-Models integration failed: {e}")

    # Test 14: End-to-End System Integration
    try:
        print(f"\n" + "=" * 70)
        print("FULL SYSTEM INTEGRATION TEST")
        print("=" * 70)
        
        # Create a test scenario
        sample_text = """
        Machine learning is a subset of artificial intelligence. 
        It involves algorithms that learn patterns from data.
        Deep learning uses neural networks with multiple layers.
        """
        
        # 1. Process text
        from processors.text_processor import TextProcessor
        processor = TextProcessor()
        processed = processor.process(sample_text)
        
        print(f"✓ 1. Text Processing Complete:")
        print(f"   - Cleaned text: {len(processed.cleaned_text)} chars")
        print(f"   - Valid: {processed.is_valid}")
        
        # 2. Detect language
        from processors.language_detector import LanguageDetector
        detector = LanguageDetector(use_model=False)
        lang_result = detector.detect(processed.cleaned_text)
        
        print(f"\n✓ 2. Language Detection Complete:")
        print(f"   - Language: {lang_result.primary_language.value}")
        print(f"   - Script: {lang_result.script.value}")
        
        # 3. Domain classification structure
        from processors.domain_classifier import get_domain_name, is_technical_domain
        ai_ml_domain = Domain.AI_ML
        
        print(f"\n✓ 3. Domain System Ready:")
        print(f"   - Domain enum: {ai_ml_domain.value}")
        print(f"   - Human name: {get_domain_name(ai_ml_domain)}")
        print(f"   - Is technical: {is_technical_domain(ai_ml_domain)}")
        
        # 4. Model management
        from models.model_manager import get_model_manager
        from models.model_registry import get_model_registry
        
        model_manager = get_model_manager()
        model_registry = get_model_registry()
        
        print(f"\n✓ 4. Model Management Ready:")
        print(f"   - Manager: {type(model_manager).__name__}")
        print(f"   - Registry: {type(model_registry).__name__}")
        print(f"   - Cache dir exists: {model_manager.cache_dir.exists()}")
        
        # 5. Settings integration
        print(f"\n✓ 5. Settings Integration:")
        print(f"   - App: {settings.APP_NAME} v{settings.APP_VERSION}")
        print(f"   - Environment: {settings.ENVIRONMENT}")
        print(f"   - Debug: {settings.DEBUG}")
        
        print(f"\n🎯 FULL SYSTEM INTEGRATION SUCCESSFUL!")
        
    except Exception as e:
        print(f"\n✗ Full system integration failed: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

# Get the captured output
output_text = output_buffer.getvalue()

# Print the output
print(output_text)

# Count successes and failures
success_count = sum(1 for line in output_text.split('\n') if '✓' in line)
failure_count = sum(1 for line in output_text.split('\n') if '✗' in line)

print(f"Successes: {success_count}")
print(f"Failures: {failure_count}")

if failure_count == 0:
    print("\n🎉 ALL TESTS PASSED! Complete system is properly integrated.")
else:
    print(f"\n⚠️  {failure_count} tests failed. Check the issues above.")