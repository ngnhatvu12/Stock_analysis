import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
from typing import Optional

class TextRewriter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    def load_model(self):
        """Load model với cài đặt tối ưu"""
        if self.is_loaded:
            return
            
        try:
            if self.model_path and os.path.exists(self.model_path):
                print(f"Loading trained model from {self.model_path}")
                try:
                    # Thử load tokenizer trước
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                    print("✅ Trained model loaded successfully!")
                except Exception as e:
                    print(f"❌ Error loading trained model: {e}")
                    print("🔄 Falling back to base model...")
                    self._load_base_model()
            else:
                self._load_base_model()
            
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._load_base_model()

    def _load_base_model(self):
        """Load base model như fallback"""
        try:
            print("Loading base ViT5 model...")
            self.tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("✅ Base model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading base model: {e}")
            self.model = None
            self.tokenizer = None

    def rewrite_rumor_text(self, text: str) -> str:
        """Viết lại văn bản với cài đặt tối ưu"""
        if not text or len(text.strip()) < 3:
            return text
        
        if not self.is_loaded:
            self.load_model()
            
        if self.model is None or self.tokenizer is None:
            return self._fallback_rewriting(text)
        
        try:
            # Tokenize với error handling
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Generate với cài đặt cân bằng
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=150,
                    num_beams=3,
                    early_stopping=True,
                    repetition_penalty=1.5,
                    length_penalty=1.0,
                    do_sample=False,
                    temperature=1.0,
                )
            
            rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._post_process(rewritten, text)
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            return self._fallback_rewriting(text)
    
    def _post_process(self, rewritten: str, original: str) -> str:
        """Hậu xử lý"""
        if not rewritten or rewritten.strip() == original.strip():
            return self._fallback_rewriting(original)
        
        rewritten = re.sub(r'\s+', ' ', rewritten).strip()
        
        if rewritten and rewritten[0].islower():
            rewritten = rewritten[0].upper() + rewritten[1:]
        
        if rewritten and rewritten[-1] not in ['.', '?', '!']:
            rewritten += '.'
        
        return rewritten
    
    def _fallback_rewriting(self, text: str) -> str:
        """Fallback thông minh"""
        # Làm sạch văn bản
        cleaned = re.sub(r'[^\w\s.,!?]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Xử lý các trường hợp đặc biệt
        if 'nghe đồn' in text.lower() or 'nghe nói' in text.lower():
            cleaned = re.sub(r'\b(nghe\s+(đồn|nói|bảo))\b\s*', '', cleaned, flags=re.IGNORECASE)
            return f"Theo thông tin đồn đoán, {cleaned}"
        elif 'hình như' in text.lower() or 'có vẻ như' in text.lower():
            cleaned = re.sub(r'\b(hình như|có vẻ như)\b\s*', '', cleaned, flags=re.IGNORECASE)
            return f"Theo nhận định, {cleaned}"
        else:
            import random
            prefixes = ["Theo thông tin,", "Có nguồn tin,", "Thông tin cho biết,"]
            prefix = random.choice(prefixes)
            return f"{prefix} {cleaned}"