from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(test_file: str, model_path: str):
    """Đánh giá mô hình trên test set"""
    from train_sentiment_model import SentimentTrainer
    
    trainer = SentimentTrainer()
    test_data = trainer.load_data(test_file)
    
    # Load model đã huấn luyện
    from transformers import pipeline
    custom_pipeline = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    true_labels = []
    pred_labels = []
    confidences = []
    
    for item in test_data:
        true_label = item['label']
        text = item['text']
        
        try:
            result = custom_pipeline(text)[0]
            pred_label = result['label']
            confidence = result['score']
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            confidences.append(confidence)
            
        except Exception as e:
            print(f"Error processing: {text} - {e}")
            continue
    
    # Báo cáo đánh giá
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=["negative", "neutral", "positive"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Phân tích confidence
    df = pd.DataFrame({
        'true_label': true_labels,
        'pred_label': pred_labels,
        'confidence': confidences
    })
    
    print("\nConfidence Analysis:")
    print(df.groupby('true_label')['confidence'].describe())
    
    return df