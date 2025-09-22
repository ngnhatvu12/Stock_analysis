import pandas as pd
import json
from typing import List, Dict
import psycopg2
from config.database import get_db_connection

class TrainingDataPreparer:
    def __init__(self):
        self.conn = get_db_connection()
    
    def extract_labeled_data(self) -> List[Dict]:
        """Trích xuất dữ liệu đã được gán nhãn từ database, bao gồm cả không có mã chứng khoán"""
        cur = self.conn.cursor()
        
        try:
            # Lấy dữ liệu từ post_summary - bao gồm cả không có mã chứng khoán
            cur.execute("""
                SELECT 
                    ps.cau_quan_trong as text,
                    ps.cam_xuc as sentiment,
                    ps.confidence_score,
                    ps.ma_chung_khoan as stock_code,
                    'post' as source
                FROM post_summary ps
                WHERE ps.cau_quan_trong IS NOT NULL 
                AND LENGTH(ps.cau_quan_trong) > 10
                AND ps.confidence_score > 0.51
                AND ps.cam_xuc IN ('TÍCH_CỰC', 'TIÊU_CỰC', 'TRUNG_TÍNH')
                
                UNION ALL
                
                SELECT 
                    rs.cau_quan_trong as text,
                    rs.cam_xuc as sentiment,
                    rs.confidence_score,
                    rs.stock_id as stock_code,
                    'reply' as source
                FROM reply_summary rs
                WHERE rs.cau_quan_trong IS NOT NULL 
                AND LENGTH(rs.cau_quan_trong) > 10
                AND rs.confidence_score > 0.51
                AND rs.cam_xuc IN ('TÍCH_CỰC', 'TIÊU_CỰC', 'TRUNG_TÍNH')
            """)
            
            data = []
            for row in cur.fetchall():
                text, sentiment, confidence, stock_code, source = row
                if text and sentiment:
                    data.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'stock_code': stock_code,  # Có thể là None
                        'source': source,
                        'has_stock': stock_code is not None  # Thêm flag để phân biệt
                    })
            
            return data
            
        except Exception as e:
            print(f"Error extracting labeled data: {e}")
            return []
        finally:
            cur.close()
    
    def create_training_dataset(self, output_path: str, min_confidence: float = 0.8, include_no_stock: bool = True):
        """Tạo dataset cho huấn luyện, bao gồm cả dữ liệu không có mã chứng khoán"""
        data = self.extract_labeled_data()
        
        # Lọc theo độ tin cậy
        filtered_data = [item for item in data if item['confidence'] >= min_confidence]
        
        # Lọc dữ liệu không có mã chứng khoán nếu cần
        if not include_no_stock:
            filtered_data = [item for item in filtered_data if item['stock_code'] is not None]
        
        # Chuyển đổi sang format phù hợp
        training_data = []
        for item in filtered_data:
            training_data.append({
                'text': item['text'],
                'label': self._map_sentiment_label(item['sentiment']),
                'stock_code': item['stock_code'],
                'has_stock': item['stock_code'] is not None
            })
        
        # Lưu thành file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Thống kê
        with_stock = sum(1 for item in training_data if item['stock_code'] is not None)
        without_stock = sum(1 for item in training_data if item['stock_code'] is None)
        
        print(f"Created training dataset with {len(training_data)} samples at {output_path}")
        print(f"With stock codes: {with_stock}, Without stock codes: {without_stock}")
        
        return training_data
    
    def create_separate_datasets(self, output_dir: str, min_confidence: float = 0.8):
        """Tạo dataset riêng cho dữ liệu có mã chứng khoán và không có mã chứng khoán"""
        data = self.extract_labeled_data()
        
        # Lọc theo độ tin cậy
        filtered_data = [item for item in data if item['confidence'] >= min_confidence]
        
        # Tách dữ liệu
        data_with_stock = [item for item in filtered_data if item['stock_code'] is not None]
        data_without_stock = [item for item in filtered_data if item['stock_code'] is None]
        
        # Lưu dataset có mã chứng khoán
        if data_with_stock:
            stock_output_path = f"{output_dir}/training_data_with_stock.jsonl"
            with open(stock_output_path, 'w', encoding='utf-8') as f:
                for item in data_with_stock:
                    f.write(json.dumps({
                        'text': item['text'],
                        'label': self._map_sentiment_label(item['sentiment']),
                        'stock_code': item['stock_code']
                    }, ensure_ascii=False) + '\n')
            print(f"Created stock dataset with {len(data_with_stock)} samples")
        
        # Lưu dataset không có mã chứng khoán
        if data_without_stock:
            no_stock_output_path = f"{output_dir}/training_data_no_stock.jsonl"
            with open(no_stock_output_path, 'w', encoding='utf-8') as f:
                for item in data_without_stock:
                    f.write(json.dumps({
                        'text': item['text'],
                        'label': self._map_sentiment_label(item['sentiment'])
                    }, ensure_ascii=False) + '\n')
            print(f"Created no-stock dataset with {len(data_without_stock)} samples")
        
        return {
            'with_stock': data_with_stock,
            'without_stock': data_without_stock
        }
    
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
        
        sentiment_counts = {'TÍCH_CỰC': 0, 'TIÊU_CỰC': 0, 'TRUNG_TÍNH': 0}
        stock_sentiment_counts = {}
        no_stock_counts = {'TÍCH_CỰC': 0, 'TIÊU_CỰC': 0, 'TRUNG_TÍNH': 0}
        
        for item in data:
            sentiment = item['sentiment']
            stock_code = item['stock_code']
            
            # Tổng số lượng theo sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Phân loại theo có mã chứng khoán hay không
            if stock_code is not None:
                if stock_code not in stock_sentiment_counts:
                    stock_sentiment_counts[stock_code] = {'TÍCH_CỰC': 0, 'TIÊU_CỰC': 0, 'TRUNG_TÍNH': 0}
                stock_sentiment_counts[stock_code][sentiment] = stock_sentiment_counts[stock_code].get(sentiment, 0) + 1
            else:
                no_stock_counts[sentiment] = no_stock_counts.get(sentiment, 0) + 1
        
        print("Overall data distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} samples")
        
        print(f"\nWithout stock codes: {sum(no_stock_counts.values())} samples")
        for sentiment, count in no_stock_counts.items():
            print(f"  {sentiment}: {count}")
        
        print(f"\nWith stock codes: {sum(sum(v.values()) for v in stock_sentiment_counts.values())} samples")
        for stock_code, counts in stock_sentiment_counts.items():
            total = sum(counts.values())
            print(f"{stock_code}: Total={total}, Positive={counts.get('TÍCH_CỰC', 0)}, "
                  f"Negative={counts.get('TIÊU_CỰC', 0)}, Neutral={counts.get('TRUNG_TÍNH', 0)}")
        
        return sentiment_counts, stock_sentiment_counts, no_stock_counts

# Sử dụng
if __name__ == "__main__":
    preparer = TrainingDataPreparer()
    
    # Tạo dataset tổng hợp bao gồm cả không có mã chứng khoán
    preparer.create_training_dataset('training_data_complete.jsonl')
    
    # Tạo dataset riêng biệt
    preparer.create_separate_datasets('./datasets')
    
    # Phân tích phân phối
    preparer.analyze_data_distribution()