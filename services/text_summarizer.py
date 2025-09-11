import requests
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import html
import unicodedata
import torch
import numpy as np
from collections import Counter

load_dotenv()

class TextSummarizer:
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        self.local_summarizer = None
        self.tokenizer = None
        self.model = None
        self._init_vietnamese_model()
    
    def _init_vietnamese_model(self):
        """Khởi tạo model VietAI/vit5 cho tiếng Việt"""
        try:
            print("Loading Vietnamese summarization model VietAI/vit5...")
            
            model_name = "VietAI/vit5-base-vietnews-summarization"  
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                print("Model loaded on GPU")
            else:
                print("Model loaded on CPU")
            
            print("VietAI/vit5 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading VietAI/vit5 model: {e}")
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Khởi tạo model fallback"""
        try:
            print("Loading fallback summarization model...")
            self.local_summarizer = pipeline(
                "summarization", 
                model="philschmid/bart-large-cnn-samsum",
                device=-1,
                torch_dtype=torch.float32
            )
            print("Fallback model loaded successfully!")
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            self.local_summarizer = None
    
    def summarize_with_vit5(self, text, max_length=150, min_length=30):
        """Sử dụng VietAI/vit5 model để tóm tắt"""
        if self.tokenizer is None or self.model is None:
            return self.summarize_with_fallback(text, max_length, min_length)
        
        try:
            cleaned_text = self._clean_text(text)
            
            if len(cleaned_text.split()) < 20:
                return self._smart_summary(text, max_length)
            
            input_text = f"vietnews: {cleaned_text}"
            
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=1.5,  
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2  
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            summary = self._post_process_summary(summary, cleaned_text)
            return self._fix_encoding(summary)
            
        except Exception as e:
            print(f"VietAI/vit5 summarization failed: {e}")
            return self.summarize_with_fallback(text, max_length, min_length)
    
    def _post_process_summary(self, summary, original_text):
        """Xử lý hậu kỳ cho bản tóm tắt"""
        summary = re.sub(r'\b(vietnews|tóm tắt|summary):?\s*', '', summary, flags=re.IGNORECASE)
        
        if len(summary.split()) < 5:
            return self._smart_summary(original_text, 100)
        
        if not summary.endswith(('.', '!', '?')):
            summary = summary.rstrip() + '.'
        
        return summary
    
    def summarize_with_fallback(self, text, max_length=100, min_length=30):
        """Sử dụng fallback model"""
        if self.local_summarizer is None:
            return self._smart_summary(text, max_length)
        
        try:
            cleaned_text = self._clean_text(text)
            
            if len(cleaned_text.split()) > 400:
                paragraphs = re.split(r'\n\n+', cleaned_text)
                if paragraphs:
                    cleaned_text = paragraphs[0] 
            
            summary = self.local_summarizer(
                cleaned_text, 
                max_length=max_length, 
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            result = self._fix_encoding(summary[0]['summary_text'])
            return result if result.strip() else self._smart_summary(text, max_length)
            
        except Exception as e:
            print(f"Fallback model failed: {e}")
            return self._smart_summary(text, max_length)
    
    def _smart_summary(self, text, max_length=100):
        cleaned_text = self._clean_text(text)
        
        if len(cleaned_text.split()) <= 8:
            return cleaned_text
        
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if not sentences:
            return cleaned_text[:max_length] + "..."
        
        important_keywords = [
            'chứng khoán', 'cổ phiếu', 'đầu tư', 'mua', 'bán', 'giá', 
            'tăng', 'giảm', 'lợi nhuận', 'khuyến nghị', 'thị trường',
            'stock', 'market', 'invest', 'buy', 'sell', 'VN-Index', 'VNINDEX',
            'cổ tức', 'dividend', 'tài chính', 'ngân hàng', 'doanh nghiệp',
            'quý', 'năm', 'triệu', 'tỷ', 'điểm', 'margin', 'tự doanh'
        ]
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            word_count = len(sentence.split())
            
            if word_count < 5 or word_count > 50:
                continue
                
            score = 0
            
            if i == 0:
                score += 3
            elif i < 3: 
                score += 1
            
            sentence_lower = sentence.lower()
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 2
                    break
            
            if re.search(r'\d', sentence):
                score += 1
            
            if 8 <= word_count <= 25:
                score += 1
            
            scored_sentences.append((sentence, score, word_count))
        
        if not scored_sentences:
            # Fallback: lấy câu đầu tiên có độ dài hợp lý
            for sentence in sentences:
                if 10 <= len(sentence.split()) <= 40:
                    return self._truncate_sentence(sentence, max_length)
            return self._truncate_sentence(cleaned_text, max_length)
        
        # Sắp xếp theo điểm số
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy các câu quan trọng nhất
        summary_parts = []
        total_length = 0
        
        for sentence, score, word_count in scored_sentences:
            if total_length + len(sentence) <= max_length:
                summary_parts.append(sentence)
                total_length += len(sentence) + 1 
            else:
                break
            
            if len(summary_parts) >= 3:
                break
        
        if not summary_parts:
            return self._truncate_sentence(sentences[0], max_length)
        
        summary = ' '.join(summary_parts)
        
        if len(summary) > max_length:
            summary = self._truncate_sentence(summary, max_length)
        
        return self._fix_encoding(summary)
    
    def _truncate_sentence(self, text, max_length):
        """Cắt câu một cách thông minh"""
        if len(text) <= max_length:
            return text
        
        # Tìm điểm cắt tự nhiên (dấu câu hoặc khoảng trắng)
        truncated = text[:max_length]
        last_punct = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        last_space = truncated.rfind(' ')
        
        if last_punct > 0 and last_punct > len(truncated) - 20:
            return truncated[:last_punct + 1]
        elif last_space > 0:
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."
    
    def _clean_text(self, text):
        """Làm sạch văn bản và fix encoding - cải tiến"""
        if not text:
            return ""
        
        text = self._fix_encoding(text)
        
        text = html.unescape(text)
        
        text = re.sub(r'http\S+', '', text)
        
        text = re.sub(r'@\w+', '', text)
        
        text = re.sub(r'#(\w+)', r'\1', text)
        
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ.,!?;:\-]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    def _fix_encoding(self, text):
        """Fix các vấn đề encoding"""
        if not text:
            return text        
        try:
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')            
            text = unicodedata.normalize('NFC', text)
            text = text.encode('utf-8', errors='ignore').decode('utf-8')          
            return text
        except:
            return text
    
    def summarize(self, text, max_length=150, min_length=30):
        try:
            return self.summarize_with_vit5(text, max_length, min_length)
        except Exception as e:
            print(f"All summarization methods failed: {e}")
            return self._smart_summary(text, max_length)

