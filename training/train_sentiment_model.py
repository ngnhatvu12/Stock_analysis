import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, load_dataset
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os

class SentimentTrainer:
    def __init__(self, model_name="vinai/phobert-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2}
        )
    
    def load_data(self, file_path: str):
        """Tải dữ liệu huấn luyện"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def preprocess_function(self, examples):
        # Tokenize text
        encoding = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

        # Map labels
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
        # Tiền xử lý dữ liệu
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
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
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
        
        # Huấn luyện
        trainer.train()
        
        # Lưu mô hình
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

# Hàm chia dữ liệu
def split_dataset(dataset, train_ratio=0.8):
    dataset = dataset.train_test_split(test_size=1-train_ratio, seed=42)
    return dataset['train'], dataset['test']

if __name__ == "__main__":
    # Chuẩn bị dữ liệu
    from data_preparation import TrainingDataPreparer
    preparer = TrainingDataPreparer()
    preparer.create_training_dataset('training_data.jsonl')
    
    # Huấn luyện
    trainer = SentimentTrainer("vinai/phobert-base")
    
    # Tải dữ liệu
    dataset = trainer.load_data('training_data.jsonl')
    train_dataset, eval_dataset = split_dataset(dataset)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Huấn luyện mô hình
    trainer.train(train_dataset, eval_dataset, output_dir="./trained_sentiment_model")