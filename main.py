import psycopg2
from config.database import get_db_connection
from services.text_summarizer import TextSummarizer
from services.stock_extractor import StockExtractor
from services.sentiment_analyzer import SentimentAnalyzer
from models.models import create_summary_table, create_reply_summary_table, create_sentiment_daily_table
import time
from datetime import datetime, timedelta
import unicodedata
import sys
import os

# Thêm đường dẫn cho thư mục training
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def process_posts_batch(batch_size=20, use_custom_model=False):
    summarizer = TextSummarizer()
    stock_extractor = StockExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Tải mô hình custom nếu được yêu cầu
    if use_custom_model and os.path.exists("./trained_sentiment_model"):
        print("Loading custom trained sentiment model...")
        sentiment_analyzer.load_custom_model("./trained_sentiment_model")
    
    create_summary_table()
    create_reply_summary_table()
    create_sentiment_daily_table()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT p.post_id, p.content, p.timestamp 
            FROM fb_post p 
            WHERE NOT EXISTS (
                SELECT 1 FROM post_summary ps WHERE ps.post_id = p.post_id
            )
            LIMIT %s
        """, (batch_size,))
        
        posts = cur.fetchall()
        
        if not posts:
            print("No posts to process.")
            return 0
        
        processed_count = 0
        start_time = time.time()
        
        for post_id, content, post_timestamp in posts:
            try:
                print(f"Processing post: {post_id}")
                
                if not content or len(content.strip()) < 10:
                    cur.execute("""
                        INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (post_id, "Content too short", None, None, "TRUNG_TÍNH", post_timestamp))
                    update_sentiment_daily_for_timestamp(post_timestamp, conn)
                    print(f"Skipped: {post_id} ")
                    continue
                
                content = unicodedata.normalize('NFC', content)
                summary = summarizer.summarize(content, max_length=100, min_length=20)
                
                stock_codes = stock_extractor.extract_stock_codes(content)
                
                cur.execute("DELETE FROM post_summary WHERE post_id = %s", (post_id,))
                
                if not stock_codes:
                    important_sentences = stock_extractor.extract_important_sentences(content, [])
                    important_sentence = important_sentences.get("GENERAL", "")
                    sentiment_text = important_sentence if important_sentence else content[:200]
                    
                    # Sử dụng custom model nếu available
                    if hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                        sentiment_result = sentiment_analyzer.analyze_with_custom_model(sentiment_text)
                    else:
                        sentiment_result = sentiment_analyzer.analyze_sentiment(sentiment_text)
                    
                    sentiment = sentiment_result["normalized_sentiment"]
                    confidence = sentiment_result["score"]
                    cur.execute("""
                    INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc, confidence_score, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                     """, (post_id, summary, None, important_sentence, sentiment, confidence, post_timestamp))
                  
                    update_sentiment_daily_for_timestamp(post_timestamp, conn)
                    print(f"Processed: {post_id} - No stock codes - Sentiment: {sentiment}")
                else:
                    # Trích xuất câu quan trọng cho mỗi mã chứng khoán
                    important_sentences = stock_extractor.extract_important_sentences(content, stock_codes)
                    
                    for stock_code in stock_codes:
                        important_sentence = important_sentences.get(stock_code, "")
                        
                        sentiment = "TRUNG_TÍNH" 
                        confidence = 0.5
                        
                        if important_sentence:
                            # Sử dụng custom model nếu available
                            if hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                                sentiment_result = sentiment_analyzer.analyze_with_custom_model(important_sentence)
                            else:
                                sentiment_result = sentiment_analyzer.analyze_sentiment(important_sentence)
                            sentiment = sentiment_result["normalized_sentiment"]
                            confidence = sentiment_result["score"]
                        else:
                            # Sử dụng custom model nếu available
                            if hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                                sentiment_result = sentiment_analyzer.analyze_with_custom_model(content)
                            else:
                                sentiment_result = sentiment_analyzer.analyze_sentiment(content)
                            sentiment = sentiment_result["normalized_sentiment"]
                            confidence = sentiment_result["score"]
                        cur.execute("""
                        INSERT INTO post_summary 
                        (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc, confidence_score, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (post_id, summary, stock_code, important_sentence, sentiment, confidence, post_timestamp))
                    
                    update_sentiment_daily_for_timestamp(post_timestamp, conn)
                    print(f"Processed: {post_id} - Stock: {stock_codes} - Multiple entries created")
                
                process_replies_for_post(post_id, summarizer, stock_extractor, sentiment_analyzer, conn, use_custom_model)
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    conn.commit()
                    print(f"Committed {processed_count} posts")
                
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                cur.execute("""
                    INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (post_id, f"Error: {str(e)[:100]}", None, None, "TRUNG_TÍNH", post_timestamp))
                update_sentiment_daily_for_timestamp(post_timestamp, conn)
                continue
        
        conn.commit()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_post = total_time / processed_count if processed_count > 0 else 0
        
        print(f"Processed {processed_count} posts in {total_time:.2f} seconds")
        print(f"Average time per post: {avg_time_per_post:.2f} seconds")
        
        return processed_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error processing batch: {e}")
        return 0
    
    finally:
        cur.close()
        conn.close()

def process_replies_for_post(post_id, summarizer, stock_extractor, sentiment_analyzer, conn, use_custom_model=False):
    """Xử lý tất cả bình luận của một post"""
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT r.reply_id, r.rely_content, p.timestamp 
            FROM fb_reply r 
            JOIN fb_post p ON r.post_id = p.post_id
            WHERE r.post_id = %s 
            AND NOT EXISTS (
                SELECT 1 FROM reply_summary rs WHERE rs.reply_id = r.reply_id
            )
        """, (post_id,))
        
        replies = cur.fetchall()
        
        if not replies:
            print(f"No replies to process for post {post_id}")
            return
        
        print(f"Processing {len(replies)} replies for post {post_id}")
        
        reply_count = 0
        
        for reply_id, reply_content, post_timestamp in replies:
            try:
                if not reply_content or len(reply_content.strip()) < 5:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (reply_id, post_id, "Content too short", None, None, "TRUNG_TINH", post_timestamp))
                    continue
                
                reply_content = unicodedata.normalize('NFC', reply_content)
                
                reply_summary = summarizer.summarize(reply_content, max_length=80, min_length=15)
                
                stock_codes = stock_extractor.extract_stock_codes(reply_content)
                
                cur.execute("DELETE FROM reply_summary WHERE reply_id = %s", (reply_id,))
                
                if not stock_codes:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (reply_id, post_id, reply_summary, None, None, "TRUNG_TINH", post_timestamp))
                    print(f"Processed reply: {reply_id} - No stock codes")
                else:
                    important_sentences = stock_extractor.extract_important_sentences(reply_content, stock_codes)
                    
                    for stock_code in stock_codes:
                        important_sentence = important_sentences.get(stock_code, "")
                        
                        sentiment = "TRUNG_TINH"  
                        confidence = 0.5
                        
                        if important_sentence:
                            # Sử dụng custom model nếu available
                            if use_custom_model and hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                                sentiment_result = sentiment_analyzer.analyze_with_custom_model(important_sentence)
                            else:
                                sentiment_result = sentiment_analyzer.analyze_sentiment(important_sentence)
                            sentiment = sentiment_result["normalized_sentiment"]
                            confidence = sentiment_result["score"]
                        
                        cur.execute("""
                            INSERT INTO reply_summary 
                            (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc, confidence_score, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (reply_id, post_id, reply_summary, stock_code, important_sentence, sentiment, confidence, post_timestamp))
                        print(f"Processed reply: {reply_id} - Stock: {stock_code} - Sentiment: {sentiment} - Confidence: {confidence:.2f}")
                
                reply_count += 1
                
                if reply_count % 10 == 0:
                    conn.commit()
                    print(f"Committed {reply_count} replies for post {post_id}")
                
            except Exception as e:
                print(f"Error processing reply {reply_id}: {e}")
                cur.execute("""
                    INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (reply_id, post_id, f"Error: {str(e)[:100]}", None, None, "TRUNG_TINH", post_timestamp))
                continue
        
        conn.commit()
        print(f"Completed processing {reply_count} replies for post {post_id}")
        
    except Exception as e:
        print(f"Error processing replies for post {post_id}: {e}")
        conn.rollback()
    finally:
        cur.close()

def update_sentiment_daily_for_timestamp(timestamp, conn):
    """Cập nhật bảng sentiment_daily cho một timestamp cụ thể"""
    cur = conn.cursor()
    
    try:
        # Đếm số lượng post distinct cho timestamp này
        cur.execute("""
            SELECT COUNT(DISTINCT post_id) 
            FROM post_summary 
            WHERE timestamp = %s
        """, (timestamp,))
        total_posts_count = cur.fetchone()[0]
        
        # Đếm số lượng sentiment entries (không phân biệt post)
        cur.execute("""
            SELECT cam_xuc, COUNT(*) 
            FROM post_summary 
            WHERE timestamp = %s 
            GROUP BY cam_xuc
        """, (timestamp,))
        
        sentiment_counts = {'TÍCH_CỰC': 0, 'TIÊU_CỰC': 0, 'TRUNG_TÍNH': 0}
        for sentiment_type, count in cur.fetchall():
            if sentiment_type in sentiment_counts:
                sentiment_counts[sentiment_type] = count
        
        # Kiểm tra xem timestamp đã tồn tại chưa
        cur.execute("SELECT COUNT(*) FROM sentiment_daily WHERE timestamp = %s", (timestamp,))
        exists = cur.fetchone()[0] > 0
        
        if exists:
            # Cập nhật dựa trên tổng số post thực tế và tổng số sentiment entries
            cur.execute("""
                UPDATE sentiment_daily 
                SET positive = %s, 
                    negative = %s,
                    neutral = %s,
                    total_posts = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE timestamp = %s
            """, (sentiment_counts['TÍCH_CỰC'], 
                  sentiment_counts['TIÊU_CỰC'], 
                  sentiment_counts['TRUNG_TÍNH'],
                  total_posts_count,
                  timestamp))
        else:
            # Tạo mới bản ghi
            cur.execute("""
                INSERT INTO sentiment_daily (timestamp, positive, negative, neutral, total_posts)
                VALUES (%s, %s, %s, %s, %s)
            """, (timestamp, 
                  sentiment_counts['TÍCH_CỰC'], 
                  sentiment_counts['TIÊU_CỰC'], 
                  sentiment_counts['TRUNG_TÍNH'],
                  total_posts_count))
        
        conn.commit()
        
    except Exception as e:
        print(f"Error updating sentiment_daily for timestamp {timestamp}: {e}")
        conn.rollback()
    finally:
        cur.close()

def get_sentiment_stats_by_timestamp(start_timestamp=None, end_timestamp=None):
    """Lấy thống kê cảm xúc theo khoảng timestamp"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        query = """
            SELECT timestamp, positive, negative, neutral, total_posts
            FROM sentiment_daily
        """
        
        params = []
        if start_timestamp and end_timestamp:
            query += " WHERE timestamp BETWEEN %s AND %s"
            params.extend([start_timestamp, end_timestamp])
        elif start_timestamp:
            query += " WHERE timestamp >= %s"
            params.append(start_timestamp)
        elif end_timestamp:
            query += " WHERE timestamp <= %s"
            params.append(end_timestamp)
        
        query += " ORDER BY timestamp"
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        return [{
            'timestamp': row[0],
            'positive': row[1],
            'negative': row[2],
            'neutral': row[3],
            'total_posts': row[4]
        } for row in results]
        
    except Exception as e:
        print(f"Error getting sentiment stats: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def retrain_sentiment_model():
    """Huấn luyện lại mô hình cảm xúc"""
    try:
        # Import các module training
        from training.data_preparation import TrainingDataPreparer
        from training.train_sentiment_model import SentimentTrainer
        
        print("Starting model retraining...")
        
        # Chuẩn bị dữ liệu
        preparer = TrainingDataPreparer()
        training_file = preparer.create_training_dataset('training_data.jsonl')
        
        # Huấn luyện
        trainer = SentimentTrainer("vinai/phobert-base")
        dataset = trainer.load_data('training_data.jsonl')
        
        if len(dataset) < 100:
            print(f"Not enough training data: {len(dataset)} samples. Need at least 100 samples.")
            return False
            
        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]

        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Evaluation samples: {len(eval_dataset)}")
        
        trainer.train(train_dataset, eval_dataset, output_dir="./trained_sentiment_model")
        
        print("Model retraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"Model retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model():
    """Đánh giá mô hình đã huấn luyện"""
    try:
        from training.evaluate_model import evaluate_model as eval_func
        
        if not os.path.exists('training_data.jsonl'):
            print("No training data found. Please run retraining first.")
            return False
            
        print("Evaluating trained model...")
        results = eval_func('training_data.jsonl', './trained_sentiment_model')
        print("Evaluation completed!")
        return True
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return False

def main():
    """Hàm main chạy một lần"""
    try:
        # Xử lý các tham số dòng lệnh
        if len(sys.argv) > 1:
            if sys.argv[1] == "--retrain":
                success = retrain_sentiment_model()
                if success:
                    print("Retraining completed. You can now use the custom model with --use-custom")
                return 0 if success else 1
                
            elif sys.argv[1] == "--evaluate":
                success = evaluate_model()
                return 0 if success else 1
                
            elif sys.argv[1] == "--use-custom":
                print(f"\n{datetime.now()} - Starting batch processing with custom model...")
                processed = process_posts_batch(batch_size=1000, use_custom_model=True)
            else:
                print(f"Unknown argument: {sys.argv[1]}")
                print("Usage: python main.py [--retrain | --evaluate | --use-custom]")
                return 1
        else:
            # Chế độ mặc định: xử lý với model gốc
            print(f"\n{datetime.now()} - Starting batch processing with default model...")
            processed = process_posts_batch(batch_size=1000)
        
        if processed == 0:
            print("No posts to process.")
        else:
            print(f"Successfully processed {processed} posts.")
            
        # Hiển thị thống kê 7 ngày gần nhất
        current_timestamp = int(time.time())
        week_ago_timestamp = current_timestamp - (7 * 24 * 60 * 60)
        stats = get_sentiment_stats_by_timestamp(week_ago_timestamp, current_timestamp)
        
        print("\nSentiment statistics for last 7 days:")
        for stat in stats:
            date_str = datetime.fromtimestamp(stat['timestamp']).strftime('%Y-%m-%d')
            print(f"{date_str}: Positive={stat['positive']}, Negative={stat['negative']}, "
                  f"Neutral={stat['neutral']}, Total={stat['total_posts']}")
            
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Tạo thư mục training nếu chưa tồn tại
    os.makedirs('training', exist_ok=True)
    main()