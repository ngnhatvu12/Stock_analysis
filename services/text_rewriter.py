# services/text_rewriter.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import os
from typing import Dict, List, Tuple

class TextRewriter:
    def __init__(self, model_name="vinai/bartpho-word"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Từ điển chuyên ngành và cách diễn đạt thay thế
        self.rewriting_patterns = {
            # Cụm từ nghe nói và cách viết lại
            'nghe nói': 'theo thông tin lan truyền',
            'nghe bảo': 'theo nguồn tin',
            'nghe đồn': 'theo thông tin đồn đoán',
            'hình như': 'có khả năng',
            'có vẻ như': 'dường như',
            
            # Từ xưng hô
            'ae': 'nhà đầu tư',
            'anh em': 'nhà đầu tư',
            'mọi người': 'các nhà đầu tư',
            'các bác': 'các nhà đầu tư',
            
            # Từ lóng thị trường
            'múc': 'mua vào',
            'gom': 'tích lũy',
            'xả hàng': 'bán ra',
            'lái': 'tổ chức',
            'gà': 'nhà đầu tư cá nhân',
            
            # Từ cảm tính
            'sml': 'giảm mạnh',
            'tèo': 'giảm sâu',
            'tím': 'giảm điểm',
            'xanh': 'tăng điểm',
            'đỏ': 'giảm điểm',
        }
        
        # Mẫu viết lại theo ngữ cảnh
        self.context_templates = {
            'probability': [
                "Theo thông tin lan truyền, khả năng {content} được đánh giá là khoảng {percentage}.",
                "Có nguồn tin cho biết tỷ lệ {content} ước tính đạt {percentage}.",
                "Theo đánh giá, xác suất {content} vào khoảng {percentage}."
            ],
            'rumor': [
                "Theo thông tin đồn đoán, {content}.",
                "Có tin lan truyền rằng {content}.",
                "Theo nguồn tin chưa kiểm chứng, {content}."
            ],
            'speculation': [
                "Theo nhận định, {content}.",
                "Có khả năng {content}.",
                "Dựa trên phân tích, {content}."
            ],
            'prediction': [
                "Theo dự báo, {content}.",
                "Triển vọng {content}.",
                "Khả năng cao {content}."
            ]
        }

    def _load_model(self):
        """Load model và tokenizer"""
        try:
            print(f"Loading text rewriting model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print("Text rewriting model loaded successfully!")
        except Exception as e:
            print(f"Error loading text rewriting model: {e}")
            self.model = None
            self.tokenizer = None

    def rewrite_rumor_text(self, text):
        """
        Viết lại tin đồn thành câu rõ ràng, trung lập, mang tính cung cấp thông tin
        """
        if not text or len(text.strip()) < 5:
            return text
        
        return self._intelligent_rewriting(text)

    def _intelligent_rewriting(self, text):
        """
        Viết lại thông minh dựa trên phân tích ngữ cảnh sâu
        """
        original_text = text
        
        # Phân tích ngữ cảnh
        context = self._analyze_context(text)
        
        # Trích xuất thông tin chính
        core_info = self._extract_core_information(text, context)
        
        # Viết lại theo ngữ cảnh
        rewritten = self._contextual_rewriting(core_info, context)
        
        # Hoàn thiện văn bản
        final_text = self._polish_rewritten_text(rewritten)
        
        # Kiểm tra chất lượng
        if not self._is_quality_improvement(final_text, original_text):
            return self._fallback_rewriting(original_text)
        
        return final_text

    def _analyze_context(self, text):
        """Phân tích ngữ cảnh của văn bản"""
        text_lower = text.lower()
        
        context = {
            'is_question': self._is_question(text_lower),
            'is_probability': self._contains_probability(text_lower),
            'is_hearsay': self._is_hearsay(text_lower),
            'is_speculation': self._is_speculation(text_lower),
            'is_prediction': self._is_prediction(text_lower),
            'has_urgency': self._has_urgency(text_lower),
            'topic': self._detect_topic(text_lower)
        }
        
        return context

    def _extract_core_information(self, text, context):
        """Trích xuất thông tin chính từ văn bản"""
        # Làm sạch văn bản
        cleaned = self._clean_text(text)
        
        # Trích xuất số liệu, phần trăm
        numbers = self._extract_numbers(cleaned)
        percentages = self._extract_percentages(cleaned)
        
        # Trích xuất chủ đề chính
        main_content = self._extract_main_content(cleaned, context)
        
        return {
            'main_content': main_content,
            'numbers': numbers,
            'percentages': percentages,
            'cleaned_text': cleaned
        }

    def _contextual_rewriting(self, core_info, context):
        """Viết lại theo ngữ cảnh cụ thể"""
        main_content = core_info['main_content']
        percentages = core_info['percentages']
        
        # Xử lý câu hỏi thành câu khẳng định
        if context['is_question']:
            return self._rewrite_question_to_statement(main_content, percentages, context)
        
        # Xử lý câu có xác suất
        elif context['is_probability'] and percentages:
            return self._rewrite_probability_statement(main_content, percentages, context)
        
        # Xử lý tin đồn
        elif context['is_hearsay']:
            return self._rewrite_hearsay_statement(main_content, context)
        
        # Xử lý suy đoán
        elif context['is_speculation']:
            return self._rewrite_speculation_statement(main_content, context)
        
        # Mặc định
        else:
            return self._rewrite_general_statement(main_content, context)

    def _rewrite_question_to_statement(self, main_content, percentages, context):
        """Viết lại câu hỏi thành câu khẳng định thông tin"""
        # Loại bỏ phần hỏi
        statement = re.sub(r'\s*(phải không|phải ko|đúng không|đúng ko|có không|có ko|\?)\s*$', '', main_content, flags=re.IGNORECASE)
        
        if percentages and len(percentages) > 0:
            percentage = percentages[0]
            remaining_percentage = 100 - int(percentage.rstrip('%'))
            
            templates = [
                "Theo thông tin lan truyền, khả năng {content} được đánh giá là khoảng {percent}. Tuy nhiên, vẫn tồn tại rủi ro nhỏ (khoảng {remaining}%) không đạt.",
                "Có nguồn tin cho biết tỷ lệ {content} ước tính đạt {percent}. Tuy nhiên, khả năng không đạt vẫn ở mức {remaining}%.",
                "Theo đánh giá, xác suất {content} vào khoảng {percent}, với rủi ro không đạt khoảng {remaining}%."
            ]
            
            import random
            template = random.choice(templates)
            return template.format(
                content=statement.strip(),
                percent=percentage,
                remaining=remaining_percentage
            )
        else:
            templates = [
                "Có thông tin cho biết {content}.",
                "Theo nguồn tin, {content}.",
                "Thông tin lan truyền cho rằng {content}."
            ]
            
            import random
            template = random.choice(templates)
            return template.format(content=statement.strip())

    def _rewrite_probability_statement(self, main_content, percentages, context):
        """Viết lại câu có xác suất"""
        if percentages and len(percentages) > 0:
            percentage = percentages[0]
            
            templates = [
                "Theo thông tin, khả năng {content} được ước tính khoảng {percent}.",
                "Có đánh giá cho rằng tỷ lệ {content} đạt mức {percent}.",
                "Theo nhận định, xác suất {content} vào khoảng {percent}."
            ]
            
            import random
            template = random.choice(templates)
            return template.format(
                content=main_content.strip(),
                percent=percentage
            )
        
        return f"Theo thông tin, {main_content.strip()}"

    def _rewrite_hearsay_statement(self, main_content, context):
        """Viết lại tin đồn"""
        # Loại bỏ cụm từ nghe nói
        content = re.sub(r'\b(nghe\s+(nói|bảo|đồn|phong thanh|đâu)|đồn\s+rằng|bảo\s+rằng)\b\s*', '', main_content, flags=re.IGNORECASE)
        
        templates = [
            "Theo thông tin lan truyền, {content}.",
            "Có tin đồn rằng {content}.",
            "Theo nguồn tin chưa kiểm chứng, {content}."
        ]
        
        import random
        template = random.choice(templates)
        return template.format(content=content.strip())

    def _rewrite_speculation_statement(self, main_content, context):
        """Viết lại câu suy đoán"""
        # Loại bỏ từ suy đoán
        content = re.sub(r'\b(hình như|có vẻ như|có lẽ|có thể|dường như)\b\s*', '', main_content, flags=re.IGNORECASE)
        
        templates = [
            "Theo nhận định, {content}.",
            "Có khả năng {content}.",
            "Dựa trên phân tích, {content}."
        ]
        
        import random
        template = random.choice(templates)
        return template.format(content=content.strip())

    def _rewrite_general_statement(self, main_content, context):
        """Viết lại câu thông thường"""
        templates = [
            "Theo thông tin, {content}.",
            "Có nguồn tin cho biết {content}.",
            "Thông tin được lan truyền cho rằng {content}."
        ]
        
        import random
        template = random.choice(templates)
        return template.format(content=main_content.strip())

    def _extract_main_content(self, text, context):
        """Trích xuất nội dung chính"""
        # Thay thế từ lóng bằng từ chuẩn
        content = text
        for slang, formal in self.rewriting_patterns.items():
            content = re.sub(r'\b' + re.escape(slang) + r'\b', formal, content, flags=re.IGNORECASE)
        
        # Chuẩn hóa cách diễn đạt
        content = self._normalize_expression(content)
        
        return content

    def _clean_text(self, text):
        """Làm sạch văn bản"""
        # Loại bỏ biểu tượng cảm xúc, từ đệm
        text = re.sub(r'[^\w\s.,!?;:()\-+%@#&*/\d]', '', text)
        text = re.sub(r'\b(ạ|vâng|dạ|ơi|nhé|nha|nè|à|ừ|uh|hả|hỉ|oi)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(kkk|haha|hehe|hihi)\b', '', text, flags=re.IGNORECASE)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _normalize_expression(self, text):
        """Chuẩn hóa cách diễn đạt"""
        replacements = {
            r'\b(tôi|mình|tao)\b': 'có nguồn tin',
            r'\b(chúng ta|chúng tôi)\b': 'các nhà đầu tư',
            r'\b(ae|anh em)\b': 'nhà đầu tư',
            r'!\s*$': '.',
            r'!!+': '.'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _extract_numbers(self, text):
        """Trích xuất số từ văn bản"""
        return re.findall(r'\b\d+(?:\.\d+)?\b', text)

    def _extract_percentages(self, text):
        """Trích xuất phần trăm từ văn bản"""
        return re.findall(r'\b\d+%\b', text)

    def _is_question(self, text_lower):
        """Kiểm tra có phải câu hỏi"""
        return any(text_lower.endswith(end) for end in ['phải không?', 'phải ko?', 'đúng không?', 'đúng ko?']) or '?' in text_lower

    def _contains_probability(self, text_lower):
        """Kiểm tra có chứa xác suất"""
        return bool(re.search(r'\b\d+%\b', text_lower))

    def _is_hearsay(self, text_lower):
        """Kiểm tra có phải tin đồn"""
        hearsay_phrases = ['nghe nói', 'nghe bảo', 'nghe đồn', 'nghe phong thanh', 'nghe đâu']
        return any(phrase in text_lower for phrase in hearsay_phrases)

    def _is_speculation(self, text_lower):
        """Kiểm tra có phải suy đoán"""
        speculation_words = ['hình như', 'có vẻ như', 'có lẽ', 'dường như']
        return any(word in text_lower for word in speculation_words)

    def _is_prediction(self, text_lower):
        """Kiểm tra có phải dự đoán"""
        prediction_words = ['sẽ', 'dự kiến', 'khả năng cao', 'triển vọng']
        return any(word in text_lower for word in prediction_words)

    def _has_urgency(self, text_lower):
        """Kiểm tra có tính khẩn cấp"""
        urgency_words = ['gấp', 'ngay', 'nhanh', 'khẩn cấp', 'phải']
        return any(word in text_lower for word in urgency_words)

    def _detect_topic(self, text_lower):
        """Phát hiện chủ đề"""
        if any(word in text_lower for word in ['nâng hạng', 'ftse', 'msci']):
            return 'market_upgrade'
        elif any(word in text_lower for word in ['cổ phiếu', 'chứng khoán', 'mã']):
            return 'stock'
        elif any(word in text_lower for word in ['lãi', 'lợi nhuận', 'doanh thu']):
            return 'earnings'
        else:
            return 'general'

    def _polish_rewritten_text(self, text):
        """Hoàn thiện văn bản đã viết lại"""
        if not text:
            return text
        
        text = text.strip()
        
        # Viết hoa chữ cái đầu
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        # Đảm bảo kết thúc bằng dấu chấm
        if text and text[-1] not in ['.', '?', '!']:
            text += '.'
        
        return text

    def _is_quality_improvement(self, rewritten, original):
        """Kiểm tra chất lượng cải thiện"""
        if not rewritten or len(rewritten.strip()) < 10:
            return False
        
        # Kiểm tra xem có thực sự viết lại không
        if rewritten.lower().replace('theo thông tin', '').replace('có nguồn tin', '').strip() == original.lower().strip():
            return False
        
        # Kiểm tra độ dài hợp lý
        if len(rewritten) < len(original) * 0.6:
            return False
        
        return True

    def _fallback_rewriting(self, text):
        """Phương pháp dự phòng khi viết lại không thành công"""
        # Làm sạch cơ bản và thêm tiền tố
        cleaned = self._clean_text(text)
        normalized = self._normalize_expression(cleaned)
        
        return f"Theo thông tin, {normalized}."
