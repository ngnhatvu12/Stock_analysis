import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from typing import Dict, List, Tuple, Any, Optional
from underthesea import word_tokenize
import unicodedata

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self._init_sentiment_models()
        
    def _init_sentiment_models(self):
        """Khởi tạo mô hình phân tích cảm xúc PhoBERT"""
        try:
            print("Loading PhoBERT sentiment analysis model...")
            
            # Sử dụng pipeline với cấu hình đúng
            try:
                self.models['phobert_sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="wonrax/phobert-base-vietnamese-sentiment",
                    tokenizer="wonrax/phobert-base-vietnamese-sentiment",
                    device=0 if torch.cuda.is_available() else -1
                )
                print(" Loaded phobert_sentiment successfully!")
            except Exception as e:
                print(f" Failed to load phobert_sentiment: {e}")
                tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
                model = AutoModelForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
                self.models['phobert_sentiment'] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                print("✓ Loaded phobert_sentiment (fallback) successfully!")
            
        except Exception as e:
            print(f"Error loading sentiment models: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Xóa URL
        text = re.sub(r"http\S+|www\S+", "", text)
        
        # Giữ mention và hashtag nhưng bỏ ký tự đặc biệt
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Giữ lại nhiều ký tự đặc biệt hơn cho biểu cảm
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ.,!?;:\-\s]', ' ', text)
        
        # Giữ lại dấu câu quan trọng cho cảm xúc
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Phân tích cảm xúc sử dụng PhoBERT"""
        if not text or len(text.strip()) < 3:
            return self._create_result("TRUNG_TÍNH", 0.5, "Text too short")
        
        cleaned_text = self.preprocess_text(text)
        
        if len(cleaned_text.strip()) < 2:
            return self._create_result("TRUNG_TÍNH", 0.5, "Text too short after cleaning")
        
        try:
            if isinstance(self.models['phobert_sentiment'], dict):
                tokenizer = self.models['phobert_sentiment']['tokenizer']
                model = self.models['phobert_sentiment']['model']
                
                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred].item()
                
                label_map = {0: 'TIÊU_CỰC', 1: 'TRUNG_TÍNH', 2: 'TÍCH_CỰC'}
                vietnamese_label = label_map.get(pred, 'TRUNG_TÍNH')
                
            else:
                result = self.models['phobert_sentiment'](cleaned_text)[0]
                
                print(f"Raw model result: {result}")
                
                label_map = {
                    'NEG': 'TIÊU_CỰC',     
                    'NEU': 'TRUNG_TÍNH',     
                    'POS': 'TÍCH_CỰC',     
                    'LABEL_0': 'TIÊU_CỰC',
                    'LABEL_1': 'TRUNG_TÍNH', 
                    'LABEL_2': 'TÍCH_CỰC'
                }
                
                label = result['label'].upper()
                vietnamese_label = label_map.get(label, 'TRUNG_TÍNH')
                confidence = result['score']
            
            # Debug output
            print(f"Analyzed: '{text}' -> {vietnamese_label} ({confidence:.4f})")
            
            return self._create_result(vietnamese_label, confidence, "phobert")
            
        except Exception as e:
            print(f"PhoBERT analysis failed: {e}")
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Phân tích cảm xúc hàng loạt"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def _create_result(self, sentiment: str, confidence: float, source: str) -> Dict[str, Any]:
        """Tạo kết quả chuẩn hóa"""
        return {
            "label": sentiment,
            "score": confidence,
            "normalized_sentiment": sentiment,
            "source": source
        }

