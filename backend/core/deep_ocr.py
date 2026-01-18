"""
Deep Learning OCR Engine with Transformer-based text recognition
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class DeepOCREngine:
    """Deep Learning OCR using Transformer models"""
    
    def __init__(self, model_name: str = "microsoft/trocr-large-printed",
                 use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            print(f"🤖 Loading Deep OCR model: {model_name}")
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Deep OCR model loaded on {self.device}")
            
        except ImportError:
            print("❌ Transformers library not installed")
            print("   Install: pip install transformers")
            self.model = None
        except Exception as e:
            print(f"❌ Failed to load Deep OCR: {e}")
            self.model = None
    
    def extract_text(self, image: np.ndarray) -> Dict:
        """Extract text using Deep OCR"""
        if self.model is None:
            return {'text': '', 'confidence': 0.0, 'engine': 'deep'}
        
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess
            pixel_values = self.processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
            
            # Calculate confidence (simplified - could use beam search scores)
            confidence = self._estimate_confidence(generated_text)
            
            return {
                'text': generated_text.strip(),
                'confidence': confidence,
                'engine': 'deep',
                'model': 'trocr'
            }
            
        except Exception as e:
            logger.error(f"Deep OCR error: {e}")
            return {'text': '', 'confidence': 0.0, 'engine': 'deep', 'error': str(e)}
    
    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence based on text characteristics"""
        if not text:
            return 0.0
        
        # Simple heuristic confidence
        confidence = 0.5  # Base confidence
        
        # Length-based confidence
        if 2 <= len(text) <= 10:
            confidence += 0.2
        
        # Alphanumeric mix confidence
        has_letters = any(c.isalpha() for c in text)
        has_digits = any(c.isdigit() for c in text)
        
        if has_letters and has_digits:
            confidence += 0.2
        
        # All uppercase (common for plates)
        if text.isupper():
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def batch_extract(self, images: list) -> list:
        """Extract text from multiple images"""
        results = []
        for img in images:
            results.append(self.extract_text(img))
        return results