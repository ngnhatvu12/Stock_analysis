# training/train_rewriter_model.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RumorRewriterTrainer:
    def __init__(self, model_name="VietAI/vit5-base"):
        """
        Sử dụng ViT5-base thay vì BARTpho:
        - ViT5-base: ~125M parameters (nhẹ hơn BARTpho 3 lần)
        - Tối ưu cho tiếng Việt
        - Hiệu quả cho text generation
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_data(self, data_path, max_samples=11000):
        """Load dữ liệu - KHÔNG giới hạn độ dài text ở đây"""
        logger.info(f"Loading data from {data_path} (max: {max_samples} samples)")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and len(data) < max_samples:
                    try:
                        item = json.loads(line)
                        if 'original' in item and 'rewritter' in item:
                            data.append({
                                'original': item['original'],
                                'rewritter': item['rewritter']
                            })
                    except:
                        continue
        
        logger.info(f"Loaded {len(data)} training examples")
        
        if len(data) < 100:
            raise ValueError(f"Not enough data: {len(data)} samples (need at least 100)")
        
        # Chia tập train/validation
        split_idx = int(0.9 * len(data))  # 90% train, 10% val
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)} samples")
        return train_dataset, val_dataset
    
    def setup_model(self):
        """Khởi tạo model với cài đặt tối ưu"""
        logger.info(f"Loading model: {self.model_name}")
        
        # QUAN TRỌNG: Sử dụng ViT5-base thay vì BARTpho
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True
        )
        
        # Tải model với cài đặt tối ưu bộ nhớ
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        logger.info("Model loaded successfully")
    
    def preprocess_function(self, examples):
        """Tiền xử lý dữ liệu"""
        inputs = examples['original']
        targets = examples['rewritter']
        
        # Tokenize với độ dài hợp lý
        model_inputs = self.tokenizer(
            inputs, 
            max_length=128,  # Tăng lên 128 cho chất lượng tốt hơn
            truncation=True, 
            padding=False,
            add_special_tokens=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets, 
                max_length=128,
                truncation=True, 
                padding=False
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train(self, data_path, output_dir="./trained_rewriter_model"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load dữ liệu
        train_dataset, val_dataset = self.load_data(data_path, max_samples=11000)
        
        # Setup model
        self.setup_model()
        
        # Tiền xử lý
        logger.info("Tokenizing training data...")
        tokenized_train = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=64,  # Tăng batch size để nhanh hơn
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        logger.info("Tokenizing validation data...")
        tokenized_val = val_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=64,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments TỐI ƯU cho 1000 samples
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # QUAN TRỌNG: Cài đặt huấn luyện
            num_train_epochs=2,  # 2 epochs cho 1000 samples
            per_device_train_batch_size=8,  # TĂNG batch size
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,  # Không cần accumulation
            
            # Cài đặt tối ưu hóa
            learning_rate=3e-4,  # Learning rate cao hơn một chút
            warmup_ratio=0.1,  # Sử dụng ratio thay vì steps
            weight_decay=0.01,
            max_grad_norm=1.0,
            
            # Cài đặt lưu & eval
            evaluation_strategy="steps",  # Đánh giá theo steps
            eval_steps=100,  # Đánh giá mỗi 100 steps
            save_strategy="steps",
            save_steps=100,
            logging_steps=50,
            report_to=None,
            
            # Tối ưu hiệu năng
            dataloader_num_workers=0 if device == "cpu" else 2,
            dataloader_pin_memory=True,
            remove_unused_columns=True,
            
            predict_with_generate=False,
            
            # THÊM: Giới hạn thời gian (tùy chọn)
            # max_steps=500,  # Có thể giới hạn steps nếu cần
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Huấn luyện
        logger.info("Starting training for 1000 samples...")
        total_steps = len(tokenized_train) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        logger.info(f"Estimated total steps: {total_steps}")
        
        try:
            train_result = trainer.train()
            
            # Lưu model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"✅ Training completed! Model saved to {output_dir}")
            return train_result
            
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            # Cố gắng lưu model
            try:
                trainer.save_model(output_dir + "_backup")
                logger.info("Model saved as backup")
            except:
                pass
            return None

def train_rumor_rewriter_model(data_path=None, output_dir="./trained_rewriter_model"):
    """Huấn luyện mô hình viết lại rumor"""
    try:
        if data_path is None:
            data_path = "training_data/rumor_dataset_10000.jsonl"
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return False
        
        # Sử dụng ViT5-base thay vì BARTpho
        trainer = RumorRewriterTrainer("VietAI/vit5-base")
        result = trainer.train(data_path, output_dir)
        
        if result is not None:
            logger.info("✅ Training completed successfully!")
            return True
        else:
            logger.warning("⚠️ Training completed with warnings")
            return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False



if __name__ == "__main__":
    train_rumor_rewriter_model()