import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from typing import Dict, List, Tuple, Any, Optional
from underthesea import word_tokenize
import unicodedata
import os
from huggingface_hub import snapshot_download

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self._init_sentiment_models()
        
    def _init_sentiment_models(self):
        """Khởi tạo mô hình phân tích cảm xúc VISOBERT"""
        try:
            print("Loading VISOBERT sentiment analysis model...")
            
            # Tải model về local trước
            model_path = self._download_model_locally("5CD-AI/Vietnamese-Sentiment-visobert")
            
            # Thử tải với pipeline
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=False,  
                    local_files_only=True
                )
                
                self.models['visobert_sentiment'] = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                print("✓ Loaded visobert_sentiment successfully with pipeline!")
                
            except Exception as e:
                print(f"Failed to load visobert_sentiment with pipeline: {e}")
                # Fallback: tải tokenizer và model riêng biệt
                print("Trying fallback loading...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=False,
                    local_files_only=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True
                )
                self.models['visobert_sentiment'] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                print("✓ Loaded visobert_sentiment (fallback) successfully!")
            
        except Exception as e:
            print(f"Error loading sentiment models: {e}")
            # Thử phương án dự phòng khác
            self._try_alternative_loading()
    
    def _download_model_locally(self, model_name: str) -> str:
        """Tải model về local để tránh các vấn đề tương thích"""
        try:
            # Tạo thư mục cache cho model
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models")
            os.makedirs(cache_dir, exist_ok=True)
            
            model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
            
            # Nếu đã tải rồi thì không tải lại
            if os.path.exists(model_path):
                print(f"Using cached model at: {model_path}")
                return model_path
            
            print(f"Downloading model {model_name} to local cache...")
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            print(f"Model downloaded to: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"Error downloading model locally: {e}")
            # Fallback: sử dụng model name trực tiếp
            return model_name
    
    def _try_alternative_loading(self):
        """Phương án dự phòng khi không thể tải model theo cách thông thường"""
        try:
            print("Trying alternative loading method...")
            
            # Sử dụng model thay thế nếu VISOBERT không hoạt động
            alternative_model = "bhadresh-savani/bert-base-uncased-emotion"
            
            tokenizer = AutoTokenizer.from_pretrained(
                alternative_model,
                use_fast=True
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                alternative_model
            )
            
            self.models['visobert_sentiment'] = {
                'tokenizer': tokenizer,
                'model': model
            }
            print("✓ Loaded alternative sentiment model successfully!")
            
        except Exception as e:
            print(f"Alternative loading also failed: {e}")
            # Tạo model giả lập để chương trình không bị crash
            self.models['visobert_sentiment'] = None
            print("Warning: Sentiment analysis model could not be loaded")
    
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
        """Phân tích cảm xúc sử dụng VISOBERT"""
        if not text or len(text.strip()) < 3:
            return self._create_result("NEUTRAL", 0.5, "Text too short")
        
        # Kiểm tra nếu model không được tải thành công
        if self.models.get('visobert_sentiment') is None:
            return self._create_result("NEUTRAL", 0.5, "Model not loaded")
        
        cleaned_text = self.preprocess_text(text)
        
        if len(cleaned_text.strip()) < 2:
            return self._create_result("NEUTRAL", 0.5, "Text too short after cleaning")
        
        try:
            if isinstance(self.models['visobert_sentiment'], dict):
                # Fallback mode
                tokenizer = self.models['visobert_sentiment']['tokenizer']
                model = self.models['visobert_sentiment']['model']
                
                inputs = tokenizer(
                    cleaned_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=256,
                    add_special_tokens=True
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred = torch.argmax(probs, dim=-1).item()
                    confidence = probs[0][pred].item()
                
                # Map labels - điều chỉnh theo model thực tế
                label_map = {
                    0: 'NEGATIVE', 
                    1: 'NEUTRAL', 
                    2: 'POSITIVE',
                    3: 'NEUTRAL',
                    4: 'POSITIVE'
                }
                english_label = label_map.get(pred, 'NEUTRAL')
                
            else:
                # Pipeline mode
                result = self.models['visobert_sentiment'](cleaned_text)[0]
                
                print(f"Raw model result: {result}")
                
                # Map labels từ output của model
                label_map = {
                    'NEG': 'NEGATIVE',     
                    'NEU': 'NEUTRAL',     
                    'POS': 'POSITIVE',     
                    'LABEL_0': 'NEGATIVE',
                    'LABEL_1': 'NEUTRAL', 
                    'LABEL_2': 'POSITIVE',
                    'negative': 'NEGATIVE',
                    'neutral': 'NEUTRAL',
                    'positive': 'POSITIVE',
                    'sadness': 'NEGATIVE',
                    'joy': 'POSITIVE',
                    'love': 'POSITIVE',
                    'anger': 'NEGATIVE',
                    'fear': 'NEGATIVE',
                    'surprise': 'NEUTRAL'
                }
                
                label = result['label']
                english_label = label_map.get(label, 'NEUTRAL')
                confidence = result['score']
            
            # Chuyển đổi sang tiếng Việt
            vietnamese_map = {
                'NEGATIVE': 'TIÊU_CỰC',
                'NEUTRAL': 'TRUNG_TÍNH', 
                'POSITIVE': 'TÍCH_CỰC'
            }
            
            vietnamese_label = vietnamese_map.get(english_label, 'TRUNG_TÍNH')
            
            # Debug output
            print(f"Analyzed: '{text[:50]}...' -> {vietnamese_label} ({confidence:.4f})")
            
            return self._create_result(vietnamese_label, confidence, "visobert")
            
        except Exception as e:
            print(f"VISOBERT analysis failed: {e}")
            # Trả về kết quả mặc định nếu có lỗi
            return self._create_result("TRUNG_TÍNH", 0.5, f"Error: {str(e)}")
    
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
    def load_custom_model(self, model_path: str):
        """Tải mô hình đã được huấn luyện"""
        try:
            print(f"Loading custom trained model from {model_path}...")
            
            self.custom_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.custom_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Tạo pipeline cho mô hình custom
            self.custom_pipeline = pipeline(
                "text-classification",
                model=self.custom_model,
                tokenizer=self.custom_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("✓ Custom model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading custom model: {e}")
            return False
    def analyze_with_custom_model(self, text: str) -> Dict[str, Any]:
        """Phân tích cảm xúc sử dụng mô hình đã huấn luyện"""
        if not hasattr(self, 'custom_pipeline') or self.custom_pipeline is None:
            return self.analyze_sentiment(text)  # Fallback to original
        
        try:
            cleaned_text = self.preprocess_text(text)
            if len(cleaned_text.strip()) < 2:
                return self._create_result("TRUNG_TÍNH", 0.5, "Text too short")
            
            result = self.custom_pipeline(cleaned_text)[0]
            
            # Map labels
            label_mapping = {
                'negative': 'TIÊU_CỰC',
                'neutral': 'TRUNG_TÍNH',
                'positive': 'TÍCH_CỰC'
            }
            
            english_label = result['label']
            vietnamese_label = label_mapping.get(english_label, 'TRUNG_TÍNH')
            confidence = result['score']
            
            return self._create_result(vietnamese_label, confidence, "custom_model")
            
        except Exception as e:
            print(f"Custom model analysis failed: {e}")
            return self.analyze_sentiment(text)  # Fallback