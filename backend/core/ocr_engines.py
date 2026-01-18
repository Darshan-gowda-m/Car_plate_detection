"""
Advanced Multi-Engine OCR System with Fixed Google Vision
"""
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
import base64
import os
import io
from pathlib import Path
import logging
import warnings

# Suppress some warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine status tracking
ENGINE_STATUS = {}

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    ENGINE_STATUS['easyocr'] = '✅ Available'
except ImportError as e:
    EASYOCR_AVAILABLE = False
    ENGINE_STATUS['easyocr'] = f'❌ Not installed: {e}'

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    ENGINE_STATUS['tesseract'] = '✅ Available'
    
    # Test Tesseract installation
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        TESSERACT_AVAILABLE = False
        ENGINE_STATUS['tesseract'] = f'❌ Tesseract not found in PATH: {e}'
        
except ImportError as e:
    TESSERACT_AVAILABLE = False
    ENGINE_STATUS['tesseract'] = f'❌ Not installed: {e}'

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GOOGLE_VISION_AVAILABLE = True
    ENGINE_STATUS['google'] = '✅ Available'
except ImportError as e:
    GOOGLE_VISION_AVAILABLE = False
    ENGINE_STATUS['google'] = f'❌ Not installed: {e}'

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .preprocessor import ImagePreprocessor


class AdvancedOCREngineManager:
    """Advanced OCR manager with multiple engines and validation"""
    
    def __init__(self, google_api_key: Optional[str] = None,
                 google_credentials_path: Optional[str] = None):
        """
        Initialize advanced OCR engines
        
        Args:
            google_api_key: Google Vision API key
            google_credentials_path: Path to Google credentials JSON
        """
        self.preprocessor = ImagePreprocessor()
        
        # Initialize engines
        self.engines = {}
        
        # 1. EasyOCR Engine
        if EASYOCR_AVAILABLE:
            try:
                # Check for GPU availability
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except:
                    gpu_available = False
                
                self.engines['easyocr'] = EasyOCREngine(gpu=gpu_available)
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.error(f"❌ EasyOCR initialization failed: {e}")
                ENGINE_STATUS['easyocr'] = f'❌ Initialization failed: {e}'
        
        # 2. Tesseract OCR Engine
        if TESSERACT_AVAILABLE:
            try:
                self.engines['tesseract'] = TesseractOCREngine()
                logger.info("✅ Tesseract OCR initialized successfully")
            except Exception as e:
                logger.error(f"❌ Tesseract initialization failed: {e}")
                ENGINE_STATUS['tesseract'] = f'❌ Initialization failed: {e}'
        
        # 3. Google Vision Engine - ALWAYS INITIALIZE EVEN WITHOUT CREDENTIALS
        self.google_api_key = google_api_key
        self.google_credentials_path = google_credentials_path
        
        try:
            self.engines['google'] = GoogleVisionOCREngine(
                api_key=google_api_key,
                credentials_path=google_credentials_path
            )
            
            if self.engines['google'].available:
                logger.info("✅ Google Vision initialized successfully")
                ENGINE_STATUS['google'] = '✅ Available'
            else:
                logger.warning("⚠️ Google Vision not available (check credentials)")
                ENGINE_STATUS['google'] = '⚠️ Check credentials'
                
        except Exception as e:
            logger.error(f"❌ Google Vision initialization failed: {e}")
            ENGINE_STATUS['google'] = f'❌ Initialization failed: {e}'
            # Still create the engine object but it won't be available
            self.engines['google'] = GoogleVisionOCREngine(
                api_key=None,
                credentials_path=None
            )
        
        # Print engine status
        print("\n🔤 OCR ENGINE STATUS")
        print("-" * 40)
        for engine, status in ENGINE_STATUS.items():
            print(f"   {engine}: {status}")
        
        available_engines = [e for e in self.engines.keys() if e != 'google' or self.engines.get('google', {}).available]
        if available_engines:
            print(f"\n✅ Available engines: {', '.join(available_engines)}")
        else:
            logger.error("❌ No OCR engines available!")
            raise RuntimeError("No OCR engines available. Please install at least one OCR engine.")
        
        # Engine priority based on accuracy
        self.engine_priority = ['google', 'tesseract', 'easyocr']
        self.engine_priority = [e for e in self.engine_priority if e in available_engines]
        
        # Performance metrics
        self.metrics = {
            'total_processed': 0,
            'engine_stats': {engine: {'count': 0, 'avg_time': 0.0, 'avg_confidence': 0.0} 
                           for engine in available_engines}
        }
        
        # Try to initialize TrOCR if available
        self.trocr_available = False
        try:
            self._initialize_trocr()
        except Exception as e:
            logger.warning(f"⚠️ TrOCR not available: {e}")
    
    def _initialize_trocr(self):
        """Initialize TrOCR model for Deep OCR"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            # Load pre-trained model
            logger.info("🤖 Loading Deep OCR model: microsoft/trocr-base-printed")
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.trocr_model = self.trocr_model.to('cuda')
                logger.info("✅ TrOCR loaded on GPU")
            else:
                logger.info("✅ TrOCR loaded on CPU")
            
            self.trocr_available = True
            logger.info("✅ Deep OCR (TrOCR) initialized successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Transformers not installed: {e}")
            self.trocr_available = False
        except Exception as e:
            logger.warning(f"⚠️ TrOCR initialization failed: {e}")
            self.trocr_available = False
    
    def extract_with_trocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using TrOCR model"""
        if not self.trocr_available:
            return "", 0.0
        
        try:
            from PIL import Image as PILImage
            import torch
            
            # Convert to PIL Image
            if len(image.shape) == 2:
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            
            # Move to device
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
                generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # For now, return medium confidence
            return generated_text.strip(), 0.7
            
        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return "", 0.0
    
    def extract_text(self, image: np.ndarray, 
                    engines: Optional[List[str]] = None,
                    preprocess: bool = True,
                    validate: bool = True,
                    use_trocr: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Extract text from image using multiple OCR engines
        
        Args:
            image: Input image
            engines: List of engines to use (None = all available)
            preprocess: Whether to preprocess image
            validate: Validate text format
            use_trocr: Whether to use TrOCR model
            
        Returns:
            Dictionary with results from each engine
        """
        self.metrics['total_processed'] += 1
        
        # Validate image
        if image is None or image.size == 0:
            logger.error("Empty image provided")
            return {}
        
        # Determine which engines to use
        if engines is None:
            engines_to_use = self.engine_priority
        else:
            engines_to_use = [e for e in engines if e in self.engines and (e != 'google' or self.engines['google'].available)]
        
        # Add TrOCR if requested and available
        if use_trocr and self.trocr_available:
            # We'll handle TrOCR separately
            pass
        
        if not engines_to_use and not (use_trocr and self.trocr_available):
            logger.error(f"No valid engines: {engines}")
            return {}
        
        # Preprocess image
        processed_images = {}
        if preprocess:
            processed_images = self.preprocessor.get_preprocessed_images(image)
            primary_image = processed_images.get('enhanced', image)
        else:
            primary_image = image
        
        # Process with each engine
        all_results = {}
        
        # Process with traditional OCR engines
        for engine_name in engines_to_use:
            try:
                engine = self.engines[engine_name]
                
                # Skip Google if not available
                if engine_name == 'google' and not engine.available:
                    logger.debug("Skipping Google Vision (not available)")
                    continue
                
                start_time = time.time()
                
                # Try enhanced version first
                text, confidence = engine.extract_text(primary_image)
                processing_time = time.time() - start_time
                
                # Try other preprocessed versions if confidence is low
                if confidence < 0.6:
                    for method, proc_image in processed_images.items():
                        if method != 'enhanced':
                            alt_text, alt_conf = engine.extract_text(proc_image)
                            if alt_conf > confidence and alt_text.strip():
                                text, confidence = alt_text, alt_conf
                
                # Validate and clean text
                if validate and text:
                    validated_text, is_valid = self._validate_plate_text(text)
                    if is_valid:
                        text = validated_text
                
                # Store result
                result = {
                    'text': text,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'is_valid': self._is_valid_plate_text(text),
                    'engine': engine_name,
                    'timestamp': time.time()
                }
                
                all_results[engine_name] = result
                
                # Update metrics
                if confidence > 0:
                    stats = self.metrics['engine_stats'][engine_name]
                    stats['count'] += 1
                    stats['avg_time'] = (stats['avg_time'] * 0.9 + processing_time * 0.1)
                    stats['avg_confidence'] = (stats['avg_confidence'] * 0.9 + confidence * 0.1)
                
                logger.debug(f"{engine_name}: '{text}' ({confidence:.3f}) in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"{engine_name} OCR error: {e}")
                all_results[engine_name] = {
                    'text': '',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'error': str(e),
                    'is_valid': False
                }
        
        # Process with TrOCR if requested
        if use_trocr and self.trocr_available:
            try:
                start_time = time.time()
                text, confidence = self.extract_with_trocr(primary_image)
                processing_time = time.time() - start_time
                
                # Validate and clean text
                if validate and text:
                    validated_text, is_valid = self._validate_plate_text(text)
                    if is_valid:
                        text = validated_text
                
                all_results['trocr'] = {
                    'text': text,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'is_valid': self._is_valid_plate_text(text),
                    'engine': 'trocr',
                    'timestamp': time.time()
                }
                
                logger.debug(f"trocr: '{text}' ({confidence:.3f}) in {processing_time:.3f}s")
                
            except Exception as e:
                logger.error(f"TrOCR error: {e}")
                all_results['trocr'] = {
                    'text': '',
                    'confidence': 0.0,
                    'processing_time': 0.0,
                    'error': str(e),
                    'is_valid': False
                }
        
        return all_results
    
    def extract_with_consensus(self, image: np.ndarray, 
                              min_confidence: float = 0.6,
                              require_agreement: int = 2,
                              use_trocr: bool = True) -> Dict[str, Any]:
        """
        Extract text using multiple engines and return consensus result
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            require_agreement: Number of engines that must agree
            use_trocr: Whether to include TrOCR in consensus
        
        Returns:
            Consensus result
        """
        results = self.extract_text(image, preprocess=True, validate=True, use_trocr=use_trocr)
        
        if not results:
            return {
                'success': False,
                'error': 'No OCR results',
                'consensus_text': '',
                'confidence': 0.0
            }
        
        # Group results by text
        text_groups = {}
        
        for engine, result in results.items():
            text = result['text']
            confidence = result['confidence']
            
            if confidence >= min_confidence and text:
                # Clean and normalize text for grouping
                cleaned_text = self._normalize_plate_text(text)
                
                if cleaned_text not in text_groups:
                    text_groups[cleaned_text] = {
                        'engines': [],
                        'total_confidence': 0,
                        'count': 0,
                        'original_texts': []
                    }
                
                text_groups[cleaned_text]['engines'].append(engine)
                text_groups[cleaned_text]['total_confidence'] += confidence
                text_groups[cleaned_text]['count'] += 1
                text_groups[cleaned_text]['original_texts'].append(text)
        
        if not text_groups:
            return {
                'success': False,
                'error': 'No confident results',
                'consensus_text': '',
                'confidence': 0.0,
                'all_results': results
            }
        
        # Find text with most agreements
        best_text = None
        best_count = 0
        best_avg_confidence = 0
        
        for text, group in text_groups.items():
            if group['count'] > best_count or (group['count'] == best_count and 
                                             (group['total_confidence'] / group['count']) > best_avg_confidence):
                best_text = text
                best_count = group['count']
                best_avg_confidence = group['total_confidence'] / group['count']
        
        # Check if we have enough agreement
        if best_count >= require_agreement:
            # Find the most common original text form
            from collections import Counter
            original_text_counter = Counter(text_groups[best_text]['original_texts'])
            most_common_original = original_text_counter.most_common(1)[0][0]
            
            return {
                'success': True,
                'consensus_text': most_common_original,
                'cleaned_text': best_text,
                'confidence': best_avg_confidence,
                'agreeing_engines': text_groups[best_text]['engines'],
                'agreement_count': best_count,
                'all_results': results
            }
        else:
            # Return best single result
            best_single = None
            best_single_conf = 0
            best_single_engine = ''
            
            for engine, result in results.items():
                if result['confidence'] > best_single_conf and result['text']:
                    best_single = result['text']
                    best_single_conf = result['confidence']
                    best_single_engine = engine
            
            return {
                'success': best_single is not None,
                'consensus_text': best_single or '',
                'confidence': best_single_conf,
                'agreeing_engines': [best_single_engine] if best_single_engine else [],
                'agreement_count': 1 if best_single else 0,
                'all_results': results
            }
    
    def _normalize_plate_text(self, text: str) -> str:
        """Normalize plate text for comparison"""
        if not text:
            return ""
        
        # Convert to uppercase, remove whitespace and special characters
        normalized = ''.join(c.upper() for c in text if c.isalnum())
        
        # Common OCR corrections
        corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B',
            ' ': '',  # Remove all spaces
            '-': '',  # Remove dashes
            '_': ''   # Remove underscores
        }
        
        # Apply corrections
        corrected = []
        for char in normalized:
            if char in corrections:
                corrected.append(corrections[char])
            else:
                corrected.append(char)
        
        return ''.join(corrected)
    
    def _validate_plate_text(self, text: str) -> Tuple[str, bool]:
        """
        Validate and clean plate text
        
        Returns:
            Tuple of (cleaned_text, is_valid)
        """
        if not text:
            return "", False
        
        # Remove unwanted characters, keep alphanumeric and spaces
        cleaned = ''.join(c for c in text if c.isalnum() or c.isspace()).strip().upper()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Check if it looks like a plate
        # Plates typically have 2-10 alphanumeric characters
        alphanumeric_only = ''.join(c for c in cleaned if c.isalnum())
        
        if 2 <= len(alphanumeric_only) <= 10:
            # Check for reasonable character distribution
            letters = sum(1 for c in alphanumeric_only if c.isalpha())
            digits = sum(1 for c in alphanumeric_only if c.isdigit())
            
            # Most plates have mix of letters and numbers
            # But allow all letters or all numbers for some cases
            if (letters > 0 and digits > 0) or len(alphanumeric_only) >= 4:
                return cleaned, True
        
        return cleaned, False
    
    def _is_valid_plate_text(self, text: str) -> bool:
        """Check if text is valid plate text"""
        if not text:
            return False
        
        # Basic validation
        cleaned = ''.join(c for c in text if c.isalnum()).upper()
        
        if 2 <= len(cleaned) <= 10:
            letters = sum(1 for c in cleaned if c.isalpha())
            digits = sum(1 for c in cleaned if c.isdigit())
            
            # Allow reasonable mixes
            return (letters + digits == len(cleaned)) and (letters > 0 or digits > 0)
        
        return False
    
    def get_metrics(self) -> Dict:
        """Get OCR performance metrics"""
        return self.metrics.copy()
    
    def get_engine_status(self) -> Dict:
        """Get engine status"""
        return ENGINE_STATUS.copy()
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines"""
        available = []
        for engine_name, engine in self.engines.items():
            if engine_name == 'google':
                if hasattr(engine, 'available') and engine.available:
                    available.append(engine_name)
            else:
                available.append(engine_name)
        return available
    
    def is_engine_available(self, engine_name: str) -> bool:
        """Check if specific engine is available"""
        if engine_name not in self.engines:
            return False
        if engine_name == 'google':
            return hasattr(self.engines[engine_name], 'available') and self.engines[engine_name].available
        return True

# Engine implementations
class EasyOCREngine:
    """EasyOCR engine implementation"""
    
    def __init__(self, gpu: bool = True):
        try:
            # First check if GPU is available
            if gpu:
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                    if not gpu_available:
                        logger.warning("GPU not available for EasyOCR, using CPU")
                except:
                    gpu_available = False
            else:
                gpu_available = False
            
            self.reader = easyocr.Reader(
                ['en'],
                gpu=gpu_available,
                model_storage_directory='models/easyocr',
                download_enabled=True
            )
            
        except Exception as e:
            logger.error(f"EasyOCR initialization error: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        try:
            # Ensure correct format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Run OCR with optimized parameters for plates
            results = self.reader.readtext(
                image,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder='beamsearch',
                batch_size=4,
                min_size=10,  # Minimum text size
                text_threshold=0.3  # Lower threshold for plates
            )
            
            # Combine results with improved logic
            texts = []
            total_confidence = 0
            count = 0
            
            for (bbox, text, confidence) in results:
                text = text.strip()
                if confidence > 0.3 and text and len(text) >= 2:  # Minimum 2 chars
                    texts.append(text)
                    total_confidence += confidence
                    count += 1
            
            combined_text = ' '.join(texts)
            avg_confidence = total_confidence / count if count > 0 else 0
            
            return combined_text, avg_confidence
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0


class TesseractOCREngine:
    """Tesseract OCR engine implementation"""
    
    def __init__(self):
        self.configs = [
            '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--psm 7 --oem 3',
            '--psm 11 --oem 3',
            '--psm 13 --oem 3'
        ]
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Tesseract"""
        try:
            from PIL import Image
            
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
            
            # Resize if too small
            width, height = pil_image.size
            if width < 100 or height < 30:
                scale = max(100.0 / width, 30.0 / height)
                new_size = (int(width * scale), int(height * scale))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Try multiple configurations
            all_texts = []
            all_confidences = []
            
            for config in self.configs:
                try:
                    # Get detailed data
                    data = pytesseract.image_to_data(
                        pil_image,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and confidence
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        conf = int(data['conf'][i])
                        
                        if conf > 40 and text and len(text) >= 2:  # Lower threshold for plates
                            all_texts.append(text)
                            all_confidences.append(conf / 100.0)
                            
                except Exception as config_error:
                    continue
            
            # Combine results
            if all_texts:
                combined_text = ' '.join(all_texts)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                return combined_text, avg_confidence
            else:
                # Fallback to simple OCR
                text = pytesseract.image_to_string(pil_image, config='--psm 8')
                return text.strip(), 0.4  # Lower default confidence
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return "", 0.0


class GoogleVisionOCREngine:
    """Google Cloud Vision engine implementation - FIXED BILLING ERROR HANDLING"""
    
    def __init__(self, api_key: Optional[str] = None,
                 credentials_path: Optional[str] = None):
        self.available = False
        self.client = None
        self.use_rest = False
        self.api_key = None
        self.billing_enabled = True  # Assume billing is enabled by default
        self.error_count = 0
        self.max_errors = 3  # Stop trying after 3 errors
        
        # Check if we should even try Google Vision
        if not api_key and not credentials_path:
            logger.info("ℹ️ Google Vision: No API key or credentials provided. Skipping.")
            ENGINE_STATUS['google'] = '❌ No credentials'
            return
        
        try:
            # Try credentials file first
            if credentials_path and Path(credentials_path).exists():
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path,
                        scopes=['https://www.googleapis.com/auth/cloud-vision']
                    )
                    self.client = vision.ImageAnnotatorClient(credentials=credentials)
                    self.available = True
                    logger.info("✅ Google Vision initialized with service account")
                    ENGINE_STATUS['google'] = '✅ Available'
                except Exception as e:
                    logger.warning(f"⚠️ Google Vision service account failed: {e}")
                    self.available = False
            
            # Try API key if credentials failed
            if not self.available and api_key:
                self.api_key = api_key
                self.api_url = "https://vision.googleapis.com/v1/images:annotate"
                self.use_rest = True
                self.available = True
                logger.info("✅ Google Vision initialized with API key")
                ENGINE_STATUS['google'] = '✅ Available'
            
            # Test connection to check billing status
            if self.available:
                self._test_billing_status()
                
        except Exception as e:
            logger.error(f"Google Vision initialization error: {e}")
            self.available = False
            ENGINE_STATUS['google'] = '❌ Initialization failed'
    
    def _test_billing_status(self):
        """Test if billing is enabled without throwing errors"""
        try:
            # Create a tiny test image
            import numpy as np
            test_image = np.zeros((10, 10, 3), dtype=np.uint8)
            test_image[2:8, 2:8] = [255, 255, 255]
            
            # Try to process it
            text, confidence = self.extract_text(test_image)
            
            if text == "" and self.error_count > 0:
                # Likely billing issue
                logger.warning("⚠️ Google Vision billing may not be enabled")
                ENGINE_STATUS['google'] = '⚠️ Check billing'
                self.billing_enabled = False
            else:
                logger.info("✅ Google Vision billing appears to be enabled")
                self.billing_enabled = True
                
        except Exception as e:
            logger.warning(f"⚠️ Google Vision billing test inconclusive: {e}")
            # Don't mark as unavailable yet, let it fail on actual use
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using Google Vision with proper error handling"""
        # Check if we should even try (too many errors or not available)
        if not self.available or self.error_count >= self.max_errors or not self.billing_enabled:
            logger.debug(f"Google Vision skipped: available={self.available}, errors={self.error_count}, billing={self.billing_enabled}")
            return "", 0.0
        
        try:
            if self.use_rest:
                return self._extract_with_rest_api(image)
            else:
                return self._extract_with_client_library(image)
                
        except Exception as e:
            error_str = str(e)
            
            # Check for billing errors
            if "BILLING_DISABLED" in error_str or "requires billing" in error_str:
                self.error_count += 1
                logger.warning(f"⚠️ Google Vision billing not enabled (error {self.error_count}/{self.max_errors})")
                ENGINE_STATUS['google'] = '⚠️ Billing disabled'
                self.billing_enabled = False
                
                # If we hit max errors, disable Google Vision completely
                if self.error_count >= self.max_errors:
                    logger.warning("🚫 Disabling Google Vision due to repeated billing errors")
                    self.available = False
                    ENGINE_STATUS['google'] = '❌ Billing disabled'
            else:
                logger.error(f"Google Vision OCR error: {e}")
            
            return "", 0.0
    
    def _extract_with_client_library(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract using client library with proper error handling"""
        import io
        
        # Convert to bytes
        success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return "", 0.0
        
        content = encoded_image.tobytes()
        
        # Create vision image
        vision_image = vision.Image(content=content)
        
        try:
            # Perform text detection with timeout
            response = self.client.text_detection(
                image=vision_image,
                timeout=5.0  # Short timeout
            )
            
            # Check for errors
            if response.error.message:
                error_msg = response.error.message
                logger.debug(f"Google Vision API error: {error_msg}")
                return "", 0.0
            
            texts = response.text_annotations
            
            if texts:
                # First annotation contains all text
                all_text = texts[0].description.strip()
                confidence = 0.8 if len(all_text) >= 2 else 0.5
                return all_text, confidence
            
            return "", 0.0
            
        except Exception as e:
            raise e
    
    def _extract_with_rest_api(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract using REST API with proper error handling"""
        if not REQUESTS_AVAILABLE:
            logger.error("Requests library not available for Google Vision REST API")
            return "", 0.0
        
        import requests
        
        # Convert to base64
        success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            return "", 0.0
        
        content = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        
        # Prepare request
        url = f"{self.api_url}?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            
            if response.status_code == 403:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', '')
                raise Exception(f"API Error 403: {error_msg}")
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API Error {response.status_code}: {error_msg}")
            
            data = response.json()
            
            if 'responses' in data and data['responses']:
                annotations = data['responses'][0]
                
                if 'error' in annotations:
                    error_msg = annotations['error'].get('message', '')
                    raise Exception(f"Annotation Error: {error_msg}")
                
                if 'textAnnotations' in annotations and annotations['textAnnotations']:
                    all_text = annotations['textAnnotations'][0]['description'].strip()
                    confidence = 0.8 if len(all_text) >= 2 else 0.5
                    return all_text, confidence
            
            return "", 0.0
            
        except requests.exceptions.Timeout:
            raise Exception("Google Vision REST API timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Google Vision REST API connection error")
        except Exception as e:
            raise e
        
OCREngineManager = AdvancedOCREngineManager