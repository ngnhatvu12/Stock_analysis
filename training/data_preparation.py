import pandas as pd
import json
from typing import List, Dict
import psycopg2
from config.database import get_db_connection

class TrainingDataPreparer:
    def __init__(self):
        self.conn = get_db_connection()
    
    def extract_labeled_data(self) -> List[Dict]:
        """Trích xuất dữ liệu đã được gán nhãn từ database"""
        cur = self.conn.cursor()
        
        try:
            # Lấy dữ liệu từ post_summary và reply_summary
            cur.execute("""
                SELECT 
                    ps.cau_quan_trong as text,
                    ps.cam_xuc as sentiment,
                    ps.confidence_score,
                    'post' as source
                FROM post_summary ps
                WHERE ps.cau_quan_trong IS NOT NULL 
                AND LENGTH(ps.cau_quan_trong) > 10
                AND ps.confidence_score > 0.51
                
                UNION ALL
                
                SELECT 
                    rs.cau_quan_trong as text,
                    rs.cam_xuc as sentiment,
                    rs.confidence_score,
                    'reply' as source
                FROM reply_summary rs
                WHERE rs.cau_quan_trong IS NOT NULL 
                AND LENGTH(rs.cau_quan_trong) > 10
                AND rs.confidence_score > 0.51
            """)
            
            data = []
            for row in cur.fetchall():
                text, sentiment, confidence, source = row
                if text and sentiment:
                    data.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'source': source
                    })
            
            return data
            
        except Exception as e:
            print(f"Error extracting labeled data: {e}")
            return []
        finally:
            cur.close()
    
    def create_training_dataset(self, output_path: str, min_confidence: float = 0.8):
        """Tạo dataset cho huấn luyện"""
        data = self.extract_labeled_data()
        
        # Lọc theo độ tin cậy
        filtered_data = [item for item in data if item['confidence'] >= min_confidence]
        
        # Chuyển đổi sang format phù hợp
        training_data = []
        for item in filtered_data:
            training_data.append({
                'text': item['text'],
                'label': self._map_sentiment_label(item['sentiment'])
            })
        
        # Lưu thành file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Created training dataset with {len(training_data)} samples at {output_path}")
        return training_data
    
    def _map_sentiment_label(self, sentiment: str) -> str:
        """Ánh xạ nhãn cảm xúc sang format chuẩn"""
        mapping = {
            'TÍCH_CỰC': 'positive',
            'TIÊU_CỰC': 'negative',
            'TRUNG_TÍNH': 'neutral'
        }
        return mapping.get(sentiment, 'neutral')
    
    def analyze_data_distribution(self):
        """Phân tích phân phối dữ liệu"""
        data = self.extract_labeled_data()
        
        sentiment_counts = {}
        for item in data:
            sentiment = item['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        print("Data distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} samples")
        
        return sentiment_counts

# Sử dụng
if __name__ == "__main__":
    preparer = TrainingDataPreparer()
    preparer.create_training_dataset('training_data.jsonl')
    preparer.analyze_data_distribution()