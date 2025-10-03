# training/evaluate_rewriter.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import json
import os
from rouge_score import rouge_scorer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RumorRewriterEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model đã huấn luyện"""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def load_test_data(self, test_data_path):
        """Load dữ liệu test"""
        logger.info(f"Loading test data from {test_data_path}")
        
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file {test_data_path} not found")
        
        data = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'original' in item and 'rewritten' in item:
                        data.append(item)
        
        logger.info(f"Loaded {len(data)} test examples")
        return data
    
    def generate_rewritten_text(self, text, max_length=512):
        """Tạo văn bản viết lại"""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=2.0,
                length_penalty=1.0
            )
        
        # Decode
        rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewritten
    
    def evaluate_rouge(self, references, predictions):
        """Đánh giá bằng ROUGE score"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = [] 
        rougeL_scores = []
        
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def evaluate(self, test_data_path, sample_size=100):
        """Đánh giá toàn diện mô hình"""
        # Load dữ liệu test
        test_data = self.load_test_data(test_data_path)
        
        # Lấy mẫu để đánh giá
        if len(test_data) > sample_size:
            import random
            test_data = random.sample(test_data, sample_size)
        
        references = []
        predictions = []
        
        logger.info("Generating predictions...")
        for i, item in enumerate(test_data):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{len(test_data)}")
            
            original = item['original']
            reference = item['rewritten']
            
            # Generate prediction
            try:
                prediction = self.generate_rewritten_text(original)
                predictions.append(prediction)
                references.append(reference)
                
                # In một số ví dụ
                if i < 5:
                    logger.info(f"\n--- Example {i+1} ---")
                    logger.info(f"Original: {original}")
                    logger.info(f"Reference: {reference}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info("-" * 50)
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        # Tính ROUGE scores
        logger.info("Calculating ROUGE scores...")
        rouge_scores = self.evaluate_rouge(references, predictions)
        
        # In kết quả
        logger.info("\n=== EVALUATION RESULTS ===")
        logger.info(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        logger.info(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        logger.info(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return {
            'rouge_scores': rouge_scores,
            'references': references,
            'predictions': predictions
        }

def evaluate_rewriter_model(model_path="./trained_rewriter_model", test_data_path=None):
    """Hàm chính để đánh giá mô hình"""
    if test_data_path is None:
        test_data_path = "training_data/rumor_dataset_10000.jsonl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return False
    
    evaluator = RumorRewriterEvaluator(model_path)
    results = evaluator.evaluate(test_data_path)
    
    return results

if __name__ == "__main__":
    evaluate_rewriter_model()