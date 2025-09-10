import re
from typing import List, Set, Dict, Any
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
import psycopg2
from config.database import get_db_connection

class StockExtractor:
    def __init__(self):
        self.common_words = self._load_common_words()
        self.stock_pattern = re.compile(r'\b([A-Z]{2,4})\b')     
        self._cache = {}
        self._cache_size = 1000      
        self.ner_model = None
        self.tokenizer = None
        self._init_ai_model()     
        self.vietnamese_stocks = self._load_stocks_from_db()
    
    def _load_stocks_from_db(self) -> Set[str]:
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            cur.execute("SELECT stock_id FROM stocks")
            stocks = cur.fetchall()
            
            stock_set = {stock[0] for stock in stocks}
            print(f"Loaded {len(stock_set)} stock codes from database")
            return stock_set
            
        except Exception as e:
            print(f"Error loading stocks from database: {e}")
            return set()
        finally:
            if conn:
                conn.close()
    
    def _init_ai_model(self):
        try:
            print("Loading AI model for stock recognition...")
            
            model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
            
            self.ner_model = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1 
            )
            print("AI model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading AI model: {e}")
            self.ner_model = None
    
    def _load_common_words(self) -> Set[str]:
        """Tải danh sách từ thông dụng cần loại bỏ"""
        return {
            'VIP', 'CEO', 'GDP', 'USD', 'VND', 'ATM', 'OK', 'HI', 'BYE', 'SH', 
            'TV', 'PC', 'IT', 'CD', 'DVD', 'USB', 'CPU', 'RAM', 'ROM', 'LAN',
            'WAN', 'MAN', 'GPS', 'LED', 'LCD', 'OLED', 'PDF', 'JPG', 'PNG',
            'GIF', 'MP3', 'MP4', 'AVI', 'MKV', 'HTML', 'CSS', 'JS', 'PHP',
            'SQL', 'API', 'URL', 'HTTP', 'HTTPS', 'FTP', 'SSH', 'SSL', 'TLS',
            'IP', 'DNS', 'VPN', 'WIFI', 'UHD','FULL','PHA','THI', 'TTCK',
            'AI', 'VR', 'AR', 'MR', 'UI', 'UX', 'CFO', 'CTO', 'COO', 'HR',
            'PR', 'QA', 'QC', 'KPI', 'ROI', 'CRM', 'ERP', 'SAP', 'HRM', 'SCM',
            'BI', 'B2B', 'B2C', 'C2C', 'O2O', 'IPO', 'M&A', 'B2G', 'G2B', 'G2C',
            'C2G', 'KQKD', 'DT','COM','NHA','TRC'
        }
    
    def _has_vietnamese_accents(self, text: str) -> bool:
        """Kiểm tra xem text có chứa dấu tiếng Việt không"""
        vietnamese_pattern = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', re.IGNORECASE)
        return bool(vietnamese_pattern.search(text))
    
    def _is_fully_uppercase_in_text(self, text: str, word: str) -> bool:
        # Tìm tất cả các vị trí xuất hiện của từ trong văn bản
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches:
            matched_text = text[match.start():match.end()]
            if matched_text.isupper():
                return True
        
        return False
    
    def _is_valid_stock_context(self, text: str, stock_code: str) -> bool:
        # Kiểm tra xem từ có được viết hoa toàn bộ không
        if not self._is_fully_uppercase_in_text(text, stock_code):
            return False
        
        # Tìm vị trí của mã trong văn bản
        pattern = re.compile(re.escape(stock_code), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches:
            start, end = match.start(), match.end()
            
            # Lấy ngữ cảnh xung quanh (20 ký tự mỗi bên)
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end].lower()
                        
            words = context.split()
            if len(words) > 1:
                for i, word in enumerate(words):
                    if stock_code.lower() in word:
                        prev_word = words[i-1] if i > 0 else ""
                        next_word = words[i+1] if i < len(words)-1 else ""
                        
                        common_vietnamese_words = {'nha', 'trc', 'pha', 'full', 'margin', 'ib', 'wed'}
                        if prev_word in common_vietnamese_words or next_word in common_vietnamese_words:
                            return False
        
        return True
    
    def extract_with_ai(self, text: str) -> List[str]:
        """Trích xuất mã chứng khoán sử dụng AI"""
        if not self.ner_model or len(text) < 10:
            return []
        
        try:
            entities = self.ner_model(text)
            
            stock_codes = set()
            for entity in entities:
                entity_text = entity['word'].upper().strip()
                
                # Bỏ qua nếu có dấu tiếng Việt
                if self._has_vietnamese_accents(entity_text):
                    continue
                
                if (2 < len(entity_text) <= 4 and 
                    entity_text.isalpha() and
                    entity_text in self.vietnamese_stocks):
                    
                    # Kiểm tra xem có phải từ thông dụng không
                    if entity_text in self.common_words:    
                        if (self._is_fully_uppercase_in_text(text, entity_text) and 
                            self._is_valid_stock_context(text, entity_text)):
                            stock_codes.add(entity_text)
                    else:
                        pattern = rf'\b{entity_text}\b'
                        if re.search(pattern, text.upper()):
                            stock_codes.add(entity_text)
            
            return list(stock_codes)
            
        except Exception as e:
            print(f"AI extraction failed: {e}")
            return []
    
    def extract_with_rules(self, text: str) -> List[str]:
        """Trích xuất mã chứng khoán sử dụng rule-based approach"""
        if not text or len(text.strip()) < 5:
            return []
        
        text_upper = text.upper()
        found_codes = set()
        
        potential_codes = re.findall(self.stock_pattern, text_upper)
        
        for code in potential_codes:
            if code in self.vietnamese_stocks:
                if code in self.common_words:                
                    if (self._is_fully_uppercase_in_text(text, code) and 
                        self._is_valid_stock_context(text, code)):
                        found_codes.add(code)
                else:
                    found_codes.add(code)
        
        return list(found_codes)
    
    def extract_stock_codes(self, text: str) -> List[str]:
        """Trích xuất mã chứng khoán từ văn bản"""
        if not text or len(text.strip()) < 5:
            return []
        
        cache_key = hash(text[:200])
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        ai_codes = self.extract_with_ai(text)
        rule_codes = self.extract_with_rules(text)
        
        all_codes = set(ai_codes + rule_codes)
        
        valid_codes = [code for code in all_codes if code in self.vietnamese_stocks]
        
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        
        self._cache[cache_key] = valid_codes
        return valid_codes
    
    def get_stock_codes_with_context(self, text: str) -> List[str]:
        """Lấy mã chứng khoán kèm ngữ cảnh (hiện tại trả về giống extract_stock_codes)"""
        return self.extract_stock_codes(text)
    
    def has_stock_mention(self, text: str) -> bool:
        """Kiểm tra xem văn bản có đề cập đến mã chứng khoán không"""
        codes = self.extract_stock_codes(text)
        return len(codes) > 0
    
    def refresh_stock_list(self):
        """Làm mới danh sách mã chứng khoán từ database"""
        self.vietnamese_stocks = self._load_stocks_from_db()
        self._cache.clear()
        print("Stock list refreshed from database")
    
    def extract_important_sentences(self, text: str, stock_codes: List[str]) -> Dict[str, str]:
        """
        Trích xuất câu quan trọng chứa mã chứng khoán
        
        Args:
            text: Văn bản gốc
            stock_codes: Danh sách mã chứng khoán đã trích xuất
            
        Returns:
            Dict với key là mã chứng khoán, value là câu quan trọng
        """
        if not text or not stock_codes:
            return {}
        
        # Tách văn bản thành các câu
        sentences = self._split_into_sentences(text)
        
        important_sentences = {}
        
        for stock_code in stock_codes:
            # Tìm câu chứa mã chứng khoán
            stock_sentences = []
            
            for sentence in sentences:
                if re.search(rf'\b{stock_code}\b', sentence, re.IGNORECASE):
                    stock_sentences.append(sentence)
            
            # Nếu có nhiều câu, chọn câu quan trọng nhất
            if stock_sentences:
                if len(stock_sentences) == 1:
                    important_sentences[stock_code] = stock_sentences[0]
                else:
                    # Chọn câu có độ dài hợp lý và chứa từ khóa quan trọng
                    scored_sentences = []
                    for sentence in stock_sentences:
                        score = self._score_sentence_importance(sentence, stock_code)
                        scored_sentences.append((sentence, score))
                    
                    scored_sentences.sort(key=lambda x: x[1], reverse=True)
                    important_sentences[stock_code] = scored_sentences[0][0]
        
        return important_sentences
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành các câu"""
        # Sử dụng regex để tách câu, xem xét các dấu câu tiếng Việt
        sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|…)\s'
        sentences = re.split(sentence_endings, text)
        
        # Lọc bỏ các câu trống
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Nếu không tách được câu, trả về toàn bộ văn bản
        if not sentences:
            return [text]
        
        return sentences
    
    def _score_sentence_importance(self, sentence: str, stock_code: str) -> float:
        """Đánh giá mức độ quan trọng của câu"""
        score = 0.0
        
        # Ưu tiên câu có độ dài vừa phải
        word_count = len(sentence.split())
        if 5 <= word_count <= 25:
            score += 2.0
        elif word_count > 25:
            score -= 1.0
        
        # Ưu tiên câu chứa từ khóa tài chính
        financial_keywords = [
            'cổ phiếu', 'chứng khoán', 'giá', 'mua', 'bán', 'tăng', 'giảm',
            'lãi', 'lỗ', 'đầu tư', 'khuyến nghị', 'thị trường', 'giao dịch',
            'cổ tức', 'dividend', 'lợi nhuận', 'doanh thu', 'tài chính'
        ]
        
        sentence_lower = sentence.lower()
        for keyword in financial_keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        # Ưu tiên câu chứa số (thường có thông tin cụ thể)
        if re.search(r'\d', sentence):
            score += 0.5
        
        # Ưu tiên câu có mã chứng khoán ở vị trí quan trọng
        if sentence.startswith(stock_code) or stock_code in sentence.split()[:3]:
            score += 1.0
        
        return score