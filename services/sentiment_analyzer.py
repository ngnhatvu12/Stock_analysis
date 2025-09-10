import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
from typing import Dict, List, Tuple, Any, Optional
import time
import requests
import os
from collections import Counter
import numpy as np
from vncorenlp import VnCoreNLP
import json

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.annotator = None
        self._init_sentiment_models()
        self._init_vncorenlp()
        
    def _init_vncorenlp(self):
        """Khởi tạo VnCoreNLP cho xử lý tiếng Việt"""
        try:
            print("Initializing VnCoreNLP for Vietnamese text processing...")    
            print("VnCoreNLP initialized successfully!")
        except Exception as e:
            print(f"VnCoreNLP initialization failed: {e}")
            self.annotator = None
    
    def _init_sentiment_models(self):
        """Khởi tạo nhiều mô hình phân tích cảm xúc"""
        try:
            print("Loading enhanced sentiment analysis models...")
            
            model_configs = [
                {
                    'name': 'phobert_sentiment',
                    'model_name': 'wonrax/phobert-base-vietnamese-sentiment',
                    'type': 'pipeline'
                },
                {
                    'name': 'vinai_phobert',
                    'model_name': 'vinai/phobert-base', 
                    'type': 'custom'
                }
            ]
            
            for config in model_configs:
                try:
                    if config['type'] == 'pipeline':
                        self.models[config['name']] = pipeline(
                            "sentiment-analysis",
                            model=config['model_name'],
                            tokenizer=config['model_name'],
                            device=0 if torch.cuda.is_available() else -1
                        )
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
                        model = AutoModelForSequenceClassification.from_pretrained(config['model_name'])
                        self.models[config['name']] = {
                            'tokenizer': tokenizer,
                            'model': model
                        }
                    
                    print(f" Loaded {config['name']} successfully!")
                    
                except Exception as e:
                    print(f" Failed to load {config['name']}: {e}")
            
            print("Available models:", list(self.models.keys()))
            
        except Exception as e:
            print(f"Error loading sentiment models: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:

        if not text or len(text.strip()) < 3:
            return self._create_result("TRUNG_TÍNH", 0.5, "Text too short")
        
        cleaned_text = self._clean_text(text)
        
        results = []
        
        transformer_result = self._analyze_with_transformers(cleaned_text)
        if transformer_result:
            results.append(transformer_result)
        
        keyword_result = self._context_aware_keyword_analysis(cleaned_text)
        results.append(keyword_result)
        
        pattern_result = self._advanced_pattern_analysis(cleaned_text)
        results.append(pattern_result)
        
        # 4. Phân tích bằng emotional intensity
        intensity_result = self._emotional_intensity_analysis(cleaned_text)
        results.append(intensity_result)
        
        # Kết hợp tất cả kết quả
        final_sentiment = self._combine_results(results, cleaned_text)
        
        return final_sentiment
    
    def _analyze_with_transformers(self, text: str) -> Optional[Dict[str, Any]]:
        """Phân tích sử dụng các model transformer"""
        if 'phobert_sentiment' not in self.models:
            return None
        
        try:
            # Giới hạn độ dài văn bản
            truncated_text = text[:512]
            
            result = self.models['phobert_sentiment'](truncated_text)[0]
            label = result['label'].upper()
            score = result['score']
            
            # Map labels to Vietnamese
            label_map = {'POSITIVE': 'TÍCH_CỰC', 'NEGATIVE': 'TIÊU_CỰC', 'NEUTRAL': 'TRUNG_TÍNH'}
            vietnamese_label = label_map.get(label, 'TRUNG_TÍNH')
            
            return {
                'sentiment': vietnamese_label,
                'confidence': score,
                'source': 'transformer'
            }
            
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return None
    
    def _context_aware_keyword_analysis(self, text: str) -> Dict[str, Any]:
        """Phân tích cảm xúc nâng cao với trọng số ngữ cảnh"""
        text_lower = text.lower()
        
        # Từ khóa tích cực với trọng số và ngữ cảnh
        positive_patterns = {
            'tốt': 2.0, 'tuyệt vời': 3.0, 'xuất sắc': 3.0, 'tăng': 1.5, 
            'lãi': 2.0, 'lợi nhuận': 2.0, 'khuyến nghị mua': 3.0,
            'mua': 1.5, 'ủng hộ': 2.0, 'tích cực': 2.0, 'lạc quan': 2.0,
            'vượt kỳ vọng': 3.0, 'ấn tượng': 2.0, 'phát triển': 1.5,
            'thành công': 2.0, 'tốt đẹp': 2.0, 'khả quan': 2.0,
            'mạnh mẽ': 1.5, 'bứt phá': 2.5, 'lên': 1.0, 'thắng': 2.0,
            'ăn': 1.5, 'ngon': 2.0, 'happy': 1.5, 'vui': 1.5,
            'sướng': 2.0, 'kiếm được': 2.0, 'lời': 2.0, 'lãi lớn': 3.0,
            'trúng lớn': 2.5, 'thành công': 2.0, 'chiến thắng': 2.0
        }
        
        # Từ khóa tiêu cực với trọng số và ngữ cảnh
        negative_patterns = {
            'xấu': 2.0, 'tệ': 2.0, 'giảm': 1.5, 'lỗ': 2.5, 'thua lỗ': 3.0,
            'khuyến nghị bán': 3.0, 'bán': 1.5, 'rút': 2.0, 'tiêu cực': 2.0,
            'bi quan': 2.0, 'thất vọng': 2.5, 'yếu kém': 2.0, 'sa sút': 2.0,
            'suy giảm': 2.0, 'khó khăn': 1.5, 'thách thức': 1.5, 'rủi ro': 2.0,
            'bất ổn': 2.0, 'lao dốc': 2.5, 'gãy trend': 2.5, 'chửi': 3.0,
            'thề': 2.0, 'đuma': 3.0, 'khốn nạn': 3.0, 'xuống': 1.5,
            'thua': 2.0, 'mất': 2.0, 'sai lầm': 2.5, 'hối hận': 2.5,
            'đau': 2.0, 'khổ': 2.0, 'khóc': 2.5, 'buồn': 2.0,
            'thất bại': 2.5, 'phá sản': 3.0, 'nợ nần': 2.5, 'trắng tay': 3.0
        }
        
        # Tính điểm với xem xét ngữ cảnh
        positive_score = self._calculate_contextual_score(text_lower, positive_patterns, True)
        negative_score = self._calculate_contextual_score(text_lower, negative_patterns, False)
        
        # Phát hiện các pattern đặc biệt
        special_patterns = [
            (r'sai.*lầm', 3.0, 'negative'),
            (r'hối.*hận', 3.0, 'negative'),
            (r'vui.*mừng', 2.5, 'positive'),
            (r'hạnh.*phúc', 2.5, 'positive'),
            (r'thắng.*lợi', 2.5, 'positive'),
            (r'mất.*tiền', 3.0, 'negative'),
            (r'lỗ.*nặng', 3.5, 'negative'),
            (r'kiếm.*tiền', 2.0, 'positive'),
            (r'lãi.*lớn', 3.0, 'positive')
        ]
        
        for pattern, weight, sentiment in special_patterns:
            if re.search(pattern, text_lower):
                if sentiment == 'positive':
                    positive_score += weight
                else:
                    negative_score += weight
        
        # Xác định cảm xúc dựa trên điểm số
        return self._determine_sentiment_from_scores(positive_score, negative_score, 'keywords')
    
    def _calculate_contextual_score(self, text: str, keywords: Dict[str, float], is_positive: bool) -> float:
        """Tính điểm với xem xét ngữ cảnh"""
        score = 0
        
        for keyword, weight in keywords.items():
            if keyword in text:
                # Tìm tất cả vị trí xuất hiện
                positions = [m.start() for m in re.finditer(keyword, text)]
                
                for pos in positions:
                    # Lấy ngữ cảnh xung quanh
                    context_start = max(0, pos - 20)
                    context_end = min(len(text), pos + len(keyword) + 20)
                    context = text[context_start:context_end]
                    
                    # Điều chỉnh trọng số dựa trên ngữ cảnh
                    contextual_weight = self._adjust_weight_by_context(context, keyword, weight, is_positive)
                    score += contextual_weight
        
        return score
    
    def _adjust_weight_by_context(self, context: str, keyword: str, base_weight: float, is_positive: bool) -> float:
        """Điều chỉnh trọng số dựa trên ngữ cảnh"""
        weight = base_weight
        
        # Tăng trọng số nếu có từ nhấn mạnh
        intensifiers = ['rất', 'quá', 'cực kỳ', 'vô cùng', 'tuyệt đối', 'hoàn toàn']
        negators = ['không', 'chẳng', 'đừng', 'đéo', 'éo']
        
        for intensifier in intensifiers:
            if intensifier in context:
                weight *= 1.5
                break
        
        # Giảm trọng số nếu có từ phủ định
        for negator in negators:
            if negator in context and any(word in context for word in [negator + ' ' + keyword, keyword + ' ' + negator]):
                weight *= -1.0  # Đảo ngược ý nghĩa
                break
        
        # Tăng trọng số nếu có dấu chấm than hoặc chữ viết hoa (biểu thị cảm xúc mạnh)
        if '!' in context or any(c.isupper() for c in context if c.isalpha()):
            weight *= 1.3
        
        return weight
    
    def _advanced_pattern_analysis(self, text: str) -> Dict[str, Any]:
        """Phân tích dựa trên các mẫu câu và cấu trúc ngữ pháp nâng cao"""
        text_lower = text.lower()
        
        # Các mẫu câu tích cực với trọng số
        positive_patterns = [
            (r'nên.*mua', 2.0),
            (r'đáng.*đầu tư', 2.5),
            (r'tiềm năng', 2.0),
            (r'cơ hội', 2.0),
            (r'triển vọng', 2.0),
            (r'khuyến.*nghị.*mua', 3.0),
            (r'tăng.*trưởng', 1.5),
            (r'lợi.*nhuận.*cao', 2.5),
            (r'sẽ.*lên', 2.0),
            (r'phát.*triển.*tốt', 2.0),
            (r'kiếm.*tiền', 2.0),
            (r'lãi.*lớn', 2.5),
            (r'thành công', 2.0)
        ]
        
        # Các mẫu câu tiêu cực với trọng số
        negative_patterns = [
            (r'nên.*bán', 2.0),
            (r'không.*nên.*mua', 2.5),
            (r'rủi.*ro', 2.0),
            (r'nguy.*hiểm', 2.0),
            (r'sợ', 1.5),
            (r'lo.*lắng', 1.5),
            (r'sai.*lầm', 2.5),
            (r'hối.*hận', 2.5),
            (r'sẽ.*xuống', 2.0),
            (r'mất.*tiền', 3.0),
            (r'thua.*lỗ', 3.0),
            (r'gãy.*trend', 2.5),
            (r'lao.*dốc', 2.5),
            (r'phá sản', 3.0),
            (r'nợ nần', 2.5)
        ]
        
        positive_score = 0
        negative_score = 0
        
        for pattern, weight in positive_patterns:
            if re.search(pattern, text_lower):
                positive_score += weight
        
        for pattern, weight in negative_patterns:
            if re.search(pattern, text_lower):
                negative_score += weight
        
        # Phát hiện cường điệu (dấu hiệu của cảm xúc mạnh)
        exaggeration_score = self._detect_exaggeration(text)
        
        if positive_score > negative_score:
            confidence = 0.6 + (exaggeration_score * 0.2)
            return {'sentiment': 'TÍCH_CỰC', 'confidence': min(confidence, 0.95), 'source': 'patterns'}
        elif negative_score > positive_score:
            confidence = 0.6 + (exaggeration_score * 0.2)
            return {'sentiment': 'TIÊU_CỰC', 'confidence': min(confidence, 0.95), 'source': 'patterns'}
        else:
            return {'sentiment': 'TRUNG_TÍNH', 'confidence': 0.5, 'source': 'patterns'}
    
    def _emotional_intensity_analysis(self, text: str) -> Dict[str, Any]:
        """Phân tích cường độ cảm xúc"""
        # Phát hiện từ ngữ cảm xúc mạnh
        strong_emotional_words = {
            'TÍCH_CỰC': ['tuyệt vời', 'xuất sắc', 'ấn tượng', 'bứt phá', 'thành công', 
                        'chiến thắng', 'hạnh phúc', 'vui mừng', 'phấn khích'],
            'TIÊU_CỰC': ['chửi thề', 'đuma', 'khốn nạn', 'thất bại', 'phá sản', 
                         'nợ nần', 'trắng tay', 'khóc', 'đau khổ']
        }
        
        text_lower = text.lower()
        positive_intensity = 0
        negative_intensity = 0
        
        for word in strong_emotional_words['TÍCH_CỰC']:
            if word in text_lower:
                positive_intensity += 1
        
        for word in strong_emotional_words['TIÊU_CỰC']:
            if word in text_lower:
                negative_intensity += 1
        
        # Phát hiện cường điệu và dấu hiệu cảm xúc mạnh
        exaggeration_score = self._detect_exaggeration(text)
        
        if positive_intensity > negative_intensity:
            confidence = 0.7 + (exaggeration_score * 0.2)
            return {'sentiment': 'TÍCH_CỰC', 'confidence': min(confidence, 0.95), 'source': 'intensity'}
        elif negative_intensity > positive_intensity:
            confidence = 0.7 + (exaggeration_score * 0.2)
            return {'sentiment': 'TIÊU_CỰC', 'confidence': min(confidence, 0.95), 'source': 'intensity'}
        else:
            return {'sentiment': 'TRUNG_TÍNH', 'confidence': 0.5, 'source': 'intensity'}
    
    def _detect_exaggeration(self, text: str) -> float:
        """Phát hiện mức độ cường điệu trong văn bản"""
        exaggeration_score = 0
        
        # Từ cường điệu
        exaggeration_words = ['rất', 'quá', 'cực kỳ', 'vô cùng', 'tuyệt đối', 
                             'hoàn toàn', 'tuyệt vời', 'kinh khủng', 'khủng khiếp']
        
        # Dấu chấm than và chữ viết hoa
        exclamation_count = text.count('!')
        capital_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Emoji cảm xúc (nếu có)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
        emoji_count = len(emoji_pattern.findall(text))
        
        # Đếm từ cường điệu
        for word in exaggeration_words:
            exaggeration_score += text.lower().count(word) * 0.3
        
        # Thêm điểm cho dấu chấm than
        exaggeration_score += min(exclamation_count * 0.2, 1.0)
        
        # Thêm điểm cho chữ viết hoa
        exaggeration_score += min(capital_ratio * 10, 2.0)
        
        # Thêm điểm cho emoji
        exaggeration_score += min(emoji_count * 0.5, 1.5)
        
        return min(exaggeration_score, 3.0) / 3.0  # Chuẩn hóa về 0-1
    
    def _determine_sentiment_from_scores(self, positive_score: float, negative_score: float, source: str) -> Dict[str, Any]:
        """Xác định cảm xúc từ điểm số"""
        total_score = positive_score + negative_score
        if total_score == 0:
            return {'sentiment': 'TRUNG_TÍNH', 'confidence': 0.5, 'source': source}
        
        confidence = abs(positive_score - negative_score) / total_score
        
        if positive_score > negative_score:
            return {'sentiment': 'TÍCH_CỰC', 'confidence': confidence, 'source': source}
        elif negative_score > positive_score:
            return {'sentiment': 'TIÊU_CỰC', 'confidence': confidence, 'source': source}
        else:
            return {'sentiment': 'TRUNG_TÍNH', 'confidence': 0.5, 'source': source}
    
    def _combine_results(self, results: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        """Kết hợp kết quả từ nhiều phương pháp"""
        if not results:
            return self._create_result("TRUNG_TÍNH", 0.5, "No results")
        
        # Đếm phiếu bầu từ các phương pháp
        votes = {'TÍCH_CỰC': 0, 'TIÊU_CỰC': 0, 'TRUNG_TÍNH': 0}
        total_confidence = 0
        
        for result in results:
            sentiment = result['sentiment']
            confidence = result.get('confidence', 0.5)
            
            votes[sentiment] += confidence
            total_confidence += confidence
        
        # Xác định cảm xúc chiến thắng
        winning_sentiment = max(votes.items(), key=lambda x: x[1])[0]
        
        # Tính confidence tổng hợp
        winning_votes = votes[winning_sentiment]
        final_confidence = winning_votes / total_confidence if total_confidence > 0 else 0.5
        
        # Hiệu chỉnh dựa trên độ dài văn bản và các yếu tố khác
        final_confidence = self._adjust_confidence(final_confidence, text, winning_sentiment)
        
        return self._create_result(winning_sentiment, final_confidence, "ensemble")
    
    def _adjust_confidence(self, confidence: float, text: str, sentiment: str) -> float:
        """Hiệu chỉnh confidence dựa trên các yếu tố bổ sung"""
        # Văn bản ngắn thường có confidence thấp hơn
        word_count = len(text.split())
        if word_count < 5:
            confidence *= 0.7
        
        # Có từ cảm xúc mạnh làm tăng confidence
        strong_emotion_words = {
            'TÍCH_CỰC': ['tuyệt vời', 'xuất sắc', 'thành công', 'chiến thắng'],
            'TIÊU_CỰC': ['chửi', 'thề', 'đuma', 'khốn', 'thất bại', 'phá sản']
        }
        
        has_strong_emotion = any(word in text.lower() for word in strong_emotion_words.get(sentiment, []))
        
        if has_strong_emotion:
            confidence = min(confidence * 1.2, 0.95)
        
        return confidence
    
    def _clean_text(self, text: str) -> str:
        """Làm sạch văn bản"""
        if not text:
            return ""
        
        # Loại bỏ URL
        text = re.sub(r'http\S+', '', text)
        
        # Loại bỏ mention và hashtag nhưng giữ text
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Loại bỏ ký tự đặc biệt nhưng giữ lại tiếng Việt
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ.,!?;:\-\s]', ' ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _create_result(self, sentiment: str, confidence: float, source: str) -> Dict[str, Any]:
        """Tạo kết quả chuẩn hóa"""
        return {
            "label": sentiment,
            "score": confidence,
            "normalized_sentiment": sentiment,
            "source": source
        }

