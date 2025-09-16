import re
import os
from typing import List, Set, Dict, Any
import time
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
import torch
import psycopg2
from config.database import get_db_connection
from summa import summarizer as textrank_summarizer
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class StockExtractor:
    def __init__(self):
        self.common_words = self._load_common_words()
        self.stock_pattern = re.compile(r'\b([A-Z]{2,4})\b')     
        self._cache = {}
        self._cache_size = 1000      
        self.ner_model = None
        self.tokenizer = None
        self.summarization_model = None
        self.vietnamese_stocks = self._load_stocks_from_db()
        self._init_ai_model()
        
        # Sử dụng NLTK thay thế cho VnCoreNLP
        print("Using NLTK for sentence splitting")
        self.rdrsegmenter = None
    
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
            print("Loading AI models for stock recognition and summarization...")
            
            # Model nhận diện thực thể (giữ nguyên)
            model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
            self.ner_model = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1 
            )
            
            # Thêm model summarization cho tiếng Việt
            summarization_model_name = "VietAI/vit5-base-vietnews-summarization"
            self.summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name)
            self.summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name)
            
            print("AI models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading AI models: {e}")
            self.ner_model = None
            self.summarization_model = None
    
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
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches:
            matched_text = text[match.start():match.end()]
            if matched_text.isupper():
                return True
        
        return False
    
    def _is_valid_stock_context(self, text: str, stock_code: str) -> bool:
        if not self._is_fully_uppercase_in_text(text, stock_code):
            return False
        
        pattern = re.compile(re.escape(stock_code), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        
        for match in matches:
            start, end = match.start(), match.end()
            
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
                
                if self._has_vietnamese_accents(entity_text):
                    continue
                
                if (2 < len(entity_text) <= 4 and 
                    entity_text.isalpha() and
                    entity_text in self.vietnamese_stocks):
                    
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
        """Lấy mã chứng khoán kèm ngữ cảnh"""
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
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành các câu với NLTK"""
        try:
            # Sử dụng NLTK để tách câu
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return [text]
            
            return sentences
        except Exception as e:
            print(f"Error splitting sentences with NLTK: {e}")
            # Fallback: sử dụng regex để tách câu
            sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|…)\s'
            sentences = re.split(sentence_endings, text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return [text]
            
            return sentences
    
    def _extract_with_textrank(self, text: str, ratio: float = 0.2) -> str:
        """Trích xuất câu quan trọng sử dụng TextRank"""
        try:
            summary = textrank_summarizer.summarize(text, ratio=ratio)
            return summary if summary else ""
        except:
            return ""
    
    def _extract_with_ai_summarization(self, text: str, max_length: int = 150) -> str:
        """Trích xuất câu quan trọng sử dụng model AI"""
        if not self.summarization_model or len(text) < 50:
            return self._extract_with_textrank(text)
        
        try:
            # Chuẩn bị input
            inputs = self.summarization_tokenizer(
                text, 
                max_length=512, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            # Generate summary
            summary_ids = self.summarization_model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode kết quả
            summary = self.summarization_tokenizer.decode(
                summary_ids[0], 
                skip_special_tokens=True
            )
            
            return summary
        except Exception as e:
            print(f"AI summarization failed: {e}")
            return self._extract_with_textrank(text)
    
    def _get_context_sentences(self, sentences: List[str], target_sentence: str, stock_code: str) -> str:
        """Lấy câu đằng trước và đằng sau câu chứa mã chứng khoán"""
        try:
            target_index = sentences.index(target_sentence)
            
            # Lấy câu đằng trước (nếu có)
            prev_sentence = sentences[target_index - 1] if target_index > 0 else ""
            
            # Lấy câu đằng sau (nếu có)
            next_sentence = sentences[target_index + 1] if target_index < len(sentences) - 1 else ""
            
            # Kết hợp thành context
            context = ""
            if prev_sentence:
                context += prev_sentence + " "
            
            context += target_sentence
            
            if next_sentence:
                context += " " + next_sentence
            
            return context.strip()
        except ValueError:
            # Nếu không tìm thấy câu trong danh sách, trả về câu gốc
            return target_sentence
    
    def extract_important_sentences(self, text: str, stock_codes: List[str]) -> Dict[str, str]:
        if not text:
            return {}
    
        sentences = self._split_into_sentences(text)
        important_sentences = {}
    
        if stock_codes:
            for stock_code in stock_codes:
                stock_sentences = []
                
                for sentence in sentences:
                    if re.search(rf'\b{stock_code}\b', sentence, re.IGNORECASE):
                        stock_sentences.append(sentence)
                
                if stock_sentences:
                    if len(stock_sentences) == 1:
                        # Lấy context cho câu duy nhất
                        context = self._get_context_sentences(sentences, stock_sentences[0], stock_code)
                        important_sentences[stock_code] = context
                    else:
                        scored_sentences = []
                        for sentence in stock_sentences:
                            score = self._score_sentence_importance(sentence, stock_code)
                            scored_sentences.append((sentence, score))
                    
                        scored_sentences.sort(key=lambda x: x[1], reverse=True)
                        # Lấy context cho câu quan trọng nhất
                        best_sentence = scored_sentences[0][0]
                        context = self._get_context_sentences(sentences, best_sentence, stock_code)
                        important_sentences[stock_code] = context
        else:
            # Không có mã chứng khoán, lấy toàn bộ context
            if len(text) > 200:
                summary = self._extract_with_ai_summarization(text)
                if summary:
                    important_sentences["GENERAL"] = summary
                else:
                    # Lấy toàn bộ text nếu summarization thất bại
                    important_sentences["GENERAL"] = text
            else:
                # Với text ngắn, lấy toàn bộ text
                important_sentences["GENERAL"] = text
    
        return important_sentences
    
    def _score_sentence_importance(self, sentence: str, stock_code: str) -> float:
        """Đánh giá mức độ quan trọng của câu"""
        score = 0.0
        
        word_count = len(sentence.split())
        if 5 <= word_count <= 25:
            score += 2.0
        elif word_count > 25:
            score -= 1.0
        
        financial_keywords = [
            'cổ phiếu', 'chứng khoán', 'giá', 'mua', 'bán', 'tăng', 'giảm',
            'lãi', 'lỗ', 'đầu tư', 'khuyến nghị', 'thị trường', 'giao dịch',
            'cổ tức', 'dividend', 'lợi nhuận', 'doanh thu', 'tài chính'
        ]
        
        sentence_lower = sentence.lower()
        for keyword in financial_keywords:
            if keyword in sentence_lower:
                score += 1.0
        
        if re.search(r'\d', sentence):
            score += 0.5

        if sentence.startswith(stock_code) or stock_code in sentence.split()[:3]:
            score += 1.0
        
        return score