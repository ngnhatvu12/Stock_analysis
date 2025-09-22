import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset, concatenate_datasets
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
from typing import Dict, List

class SentimentTrainer:
    def __init__(self, model_name="vinai/phobert-base", include_stock_info=False):
        self.model_name = model_name
        self.include_stock_info = include_stock_info
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Cấu hình mô hình
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        )
        
        # Nếu bao gồm thông tin mã chứng khoán, thêm special tokens
        if include_stock_info:
            stock_tokens = ["[STOCK]", "[/STOCK]"] + [f"[{code}]" for code in self.get_common_stock_codes()]
            self.tokenizer.add_tokens(stock_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def get_common_stock_codes(self) -> List[str]:
        """Lấy danh sách mã chứng khoán phổ biến từ database"""
        # Bạn có thể implement hàm này để lấy mã chứng khoán từ database
        # Tạm thời trả về danh sách rỗng
        return []
    
    def load_data(self, file_path: str):
        """Tải dữ liệu huấn luyện"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        
        return Dataset.from_list(data)
    
    def load_multiple_datasets(self, file_paths: List[str]):
        """Tải và kết hợp nhiều dataset"""
        datasets = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                dataset = self.load_data(file_path)
                datasets.append(dataset)
        
        if not datasets:
            raise ValueError("No valid datasets found")
        
        return concatenate_datasets(datasets)
    
    def preprocess_function(self, examples):
        """Tiền xử lý dữ liệu với thông tin mã chứng khoán"""
        texts = examples["text"]
        
        # Thêm thông tin mã chứng khoán vào text nếu có và được yêu cầu
        if self.include_stock_info and "stock_code" in examples:
            stock_codes = examples["stock_code"]
            processed_texts = []
            for text, stock_code in zip(texts, stock_codes):
                if stock_code:
                    processed_text = f"[STOCK]{stock_code}[/STOCK] {text}"
                else:
                    processed_text = text
                processed_texts.append(processed_text)
            texts = processed_texts
        
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256
        )

        label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        encoding["labels"] = [label_mapping[label] for label in examples["label"]]

        return encoding
    
    def compute_metrics(self, eval_pred):
        """Tính toán metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, eval_dataset=None, output_dir="./sentiment_model"):
        """Huấn luyện mô hình"""
        tokenized_train = train_dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            tokenized_eval = eval_dataset.map(
                self.preprocess_function, 
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Thiết lập training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir='./logs',
            logging_steps=10,
            report_to=None
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval if eval_dataset else None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        trainer.train()
        
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

def split_dataset(dataset, train_ratio=0.8):
    dataset = dataset.train_test_split(test_size=1-train_ratio, seed=42)
    return dataset['train'], dataset['test']

def train_comprehensive_model():
    """Huấn luyện mô hình tổng hợp bao gồm cả dữ liệu không có mã chứng khoán"""
    from data_preparation import TrainingDataPreparer
    
    preparer = TrainingDataPreparer()
    
    # Tạo dataset tổng hợp
    training_data = preparer.create_training_dataset('training_data_complete.jsonl')
    
    # Huấn luyện mô hình
    trainer = SentimentTrainer("vinai/phobert-base", include_stock_info=True)
    dataset = trainer.load_data('training_data_complete.jsonl')
    
    if len(dataset) < 100:
        print(f"Not enough training data: {len(dataset)} samples. Need at least 100 samples.")
        return False
    
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    trainer.train(train_dataset, eval_dataset, output_dir="./trained_sentiment_model_complete")
    
    return True

if __name__ == "__main__":
    # Huấn luyện mô hình tổng hợp
    train_comprehensive_model()