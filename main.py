import psycopg2
from config.database import get_db_connection
from services.stock_extractor import StockExtractor
from services.sentiment_analyzer import SentimentAnalyzer
from models.models import create_summary_table, create_reply_summary_table
import time
from datetime import datetime, timedelta
import unicodedata
import sys
import os

# Thêm đường dẫn cho thư mục training
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

def process_posts_batch(batch_size=20, use_custom_model=False, source='facebook', last_24h_only=False):
    stock_extractor = StockExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Tải mô hình custom nếu được yêu cầu
    if use_custom_model and os.path.exists("./trained_sentiment_model_complete"):
        print("Loading custom trained sentiment model...")
        sentiment_analyzer.load_custom_model("./trained_sentiment_model_complete")
    
    create_summary_table()
    create_reply_summary_table()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        current_timestamp = int(time.time())
        twenty_four_hours_ago = current_timestamp - (24 * 60 * 60)
        
        if source == 'facebook':
            # Xử lý dữ liệu từ Facebook
            if last_24h_only:
                cur.execute("""
                    SELECT p.post_id, p.content, p.timestamp 
                    FROM fb_post p 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = p.post_id::text AND ps.source = 'facebook'
                    )
                    AND p.timestamp >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT p.post_id, p.content, p.timestamp 
                    FROM fb_post p 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = p.post_id::text AND ps.source = 'facebook'
                    )
                    LIMIT %s
                """, (batch_size,))
        elif source == 'youtube':
            # Xử lý dữ liệu từ YouTube
            if last_24h_only:
                cur.execute("""
                    SELECT y.post_id, y.post_sentence, EXTRACT(epoch FROM y.post_at)::bigint
                    FROM yt_post_summary y 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = y.post_id::text AND ps.source = 'youtube'
                    )
                    AND EXTRACT(epoch FROM y.post_at) >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT y.post_id, y.post_sentence, EXTRACT(epoch FROM y.post_at)::bigint
                    FROM yt_post_summary y 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = y.post_id::text AND ps.source = 'youtube'
                    )
                    LIMIT %s
                """, (batch_size,))
        elif source == 'fireant':
            # Xử lý dữ liệu từ FireAnt
            if last_24h_only:
                cur.execute("""
                    SELECT f.post_id, f.original_content, EXTRACT(epoch FROM f.date)::bigint
                    FROM fireant_posts f 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = f.post_id::text AND ps.source = 'fireant'
                    )
                    AND EXTRACT(epoch FROM f.date) >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT f.post_id, f.original_content, EXTRACT(epoch FROM f.date)::bigint
                    FROM fireant_posts f 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM post_summary ps WHERE ps.post_id = f.post_id::text AND ps.source = 'fireant'
                    )
                    LIMIT %s
                """, (batch_size,))
        
        posts = cur.fetchall()
        
        if not posts:
            time_constraint = " from last 24 hours" if last_24h_only else ""
            print(f"No {source} posts{time_constraint} to process.")
            return 0
        
        processed_count = 0
        start_time = time.time()
        
        for post_id, content, post_timestamp in posts:
            try:
                print(f"Processing {source} post: {post_id}")
                
                # Đảm bảo post_id là string
                post_id_str = str(post_id)
                
                if not content or len(content.strip()) < 10:
                    cur.execute("""
                        INSERT INTO post_summary (post_id, ma_chung_khoan, cau_quan_trong, cam_xuc, timestamp, source)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (post_id_str, None, None, "TRUNG_TÍNH", post_timestamp, source))
                    print(f"Skipped: {post_id_str} ")
                    continue
                
                content = unicodedata.normalize('NFC', content)
                
                stock_codes = stock_extractor.extract_stock_codes(content)
                
                cur.execute("DELETE FROM post_summary WHERE post_id = %s AND source = %s", (post_id_str, source))
                
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
                    INSERT INTO post_summary (post_id, ma_chung_khoan, cau_quan_trong, cam_xuc, confidence_score, timestamp, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                     """, (post_id_str, None, important_sentence, sentiment, confidence, post_timestamp, source))
                  
                    print(f"Processed: {post_id_str} - No stock codes - Sentiment: {sentiment}")
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
                        (post_id, ma_chung_khoan, cau_quan_trong, cam_xuc, confidence_score, timestamp, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (post_id_str, stock_code, important_sentence, sentiment, confidence, post_timestamp, source))
                    
                    print(f"Processed: {post_id_str} - Stock: {stock_codes} - Multiple entries created")
                
                # Chỉ xử lý replies cho Facebook posts
                if source == 'facebook':
                    process_replies_for_post(post_id_str, stock_extractor, sentiment_analyzer, conn, use_custom_model)
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    conn.commit()
                    print(f"Committed {processed_count} posts")
                
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                cur.execute("""
                    INSERT INTO post_summary (post_id, ma_chung_khoan, cau_quan_trong, cam_xuc, timestamp, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (post_id_str, None, None, "TRUNG_TÍNH", post_timestamp, source))
                continue
        
        conn.commit()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_post = total_time / processed_count if processed_count > 0 else 0
        
        time_constraint = " from last 24 hours" if last_24h_only else ""
        print(f"Processed {processed_count} {source} posts{time_constraint} in {total_time:.2f} seconds")
        print(f"Average time per post: {avg_time_per_post:.2f} seconds")
        
        return processed_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error processing batch: {e}")
        return 0
    
    finally:
        cur.close()
        conn.close()

def process_replies_for_post(post_id, stock_extractor, sentiment_analyzer, conn, use_custom_model=False):
    """Xử lý tất cả bình luận của một post (chỉ cho Facebook)"""
    cur = conn.cursor()
    
    try:
        # Đảm bảo post_id là string
        post_id_str = str(post_id)
        
        cur.execute("""
            SELECT r.reply_id, r.rely_content, p.timestamp 
            FROM fb_reply r 
            JOIN fb_post p ON r.post_id = p.post_id
            WHERE r.post_id = %s 
            AND NOT EXISTS (
                SELECT 1 FROM reply_summary rs WHERE rs.reply_id = r.reply_id::text
            )
        """, (post_id_str,))
        
        replies = cur.fetchall()
        
        if not replies:
            print(f"No replies to process for post {post_id_str}")
            return
        
        print(f"Processing {len(replies)} replies for post {post_id_str}")
        
        reply_count = 0
        
        for reply_id, reply_content, post_timestamp in replies:
            try:
                # Đảm bảo reply_id là string
                reply_id_str = str(reply_id)
                
                if not reply_content or len(reply_content.strip()) < 5:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, stock_id, cau_quan_trong, cam_xuc, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (reply_id_str, post_id_str, None, None, "TRUNG_TINH", post_timestamp))
                    continue
                
                reply_content = unicodedata.normalize('NFC', reply_content)
                
                stock_codes = stock_extractor.extract_stock_codes(reply_content)
                
                cur.execute("DELETE FROM reply_summary WHERE reply_id = %s", (reply_id_str,))
                
                if not stock_codes:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, stock_id, cau_quan_trong, cam_xuc, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (reply_id_str, post_id_str, None, None, "TRUNG_TINH", post_timestamp))
                    print(f"Processed reply: {reply_id_str} - No stock codes")
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
                            (reply_id, post_id, stock_id, cau_quan_trong, cam_xuc, confidence_score, timestamp)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (reply_id_str, post_id_str, stock_code, important_sentence, sentiment, confidence, post_timestamp))
                        print(f"Processed reply: {reply_id_str} - Stock: {stock_code} - Sentiment: {sentiment} - Confidence: {confidence:.2f}")
                
                reply_count += 1
                
                if reply_count % 10 == 0:
                    conn.commit()
                    print(f"Committed {reply_count} replies for post {post_id_str}")
                
            except Exception as e:
                print(f"Error processing reply {reply_id}: {e}")
                cur.execute("""
                    INSERT INTO reply_summary (reply_id, post_id, stock_id, cau_quan_trong, cam_xuc, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (reply_id_str, post_id_str, None, None, "TRUNG_TINH", post_timestamp))
                continue
        
        conn.commit()
        print(f"Completed processing {reply_count} replies for post {post_id_str}")
        
    except Exception as e:
        print(f"Error processing replies for post {post_id}: {e}")
        conn.rollback()
    finally:
        cur.close()

def retrain_sentiment_model():
    """Huấn luyện lại mô hình cảm xúc với dữ liệu đầy đủ"""
    try:
        # Import các module training
        from training.data_preparation import TrainingDataPreparer
        from training.train_sentiment_model import SentimentTrainer, train_comprehensive_model
        
        print("Starting comprehensive model retraining...")
        
        # Chuẩn bị dữ liệu đầy đủ (bao gồm cả không có mã chứng khoán)
        preparer = TrainingDataPreparer()
        training_data = preparer.create_training_dataset('training_data_complete.jsonl')
        
        # Huấn luyện mô hình tổng hợp
        success = train_comprehensive_model()
        
        if success:
            print("Comprehensive model retraining completed successfully!")
        else:
            print("Model retraining failed due to insufficient data")
        
        return success
        
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
                # Xử lý Facebook posts
                processed_fb = process_posts_batch(batch_size=1000, use_custom_model=True, source='facebook')
                # Xử lý YouTube posts
                processed_yt = process_posts_batch(batch_size=1000, use_custom_model=True, source='youtube')
                # Xử lý FireAnt posts
                processed_fa = process_posts_batch(batch_size=1000, use_custom_model=True, source='fireant')
                processed = processed_fb + processed_yt + processed_fa
            elif sys.argv[1] == "--youtube-only":
                print(f"\n{datetime.now()} - Processing YouTube posts only...")
                processed = process_posts_batch(batch_size=1000, source='youtube')
            elif sys.argv[1] == "--fireant-only":
                print(f"\n{datetime.now()} - Processing FireAnt posts only...")
                processed = process_posts_batch(batch_size=1000, source='fireant')
            elif sys.argv[1] == "--last-24h":
                print(f"\n{datetime.now()} - Processing posts from last 24 hours with custom model...")
                # Xử lý Facebook posts từ 24h qua
                processed_fb = process_posts_batch(batch_size=1000, use_custom_model=True, source='facebook', last_24h_only=True)
                # Xử lý YouTube posts từ 24h qua
                processed_yt = process_posts_batch(batch_size=1000, use_custom_model=True, source='youtube', last_24h_only=True)
                # Xử lý FireAnt posts từ 24h qua
                processed_fa = process_posts_batch(batch_size=1000, use_custom_model=True, source='fireant', last_24h_only=True)
                processed = processed_fb + processed_yt + processed_fa
            else:
                print(f"Unknown argument: {sys.argv[1]}")
                print("Usage: python main.py [--retrain | --evaluate | --use-custom | --youtube-only | --fireant-only | --last-24h]")
                return 1
        else:
            # Chế độ mặc định: xử lý với model gốc
            print(f"\n{datetime.now()} - Starting batch processing with default model...")
            # Xử lý Facebook posts
            processed_fb = process_posts_batch(batch_size=1000, source='facebook')
            # Xử lý YouTube posts
            processed_yt = process_posts_batch(batch_size=1000, source='youtube')
            # Xử lý FireAnt posts
            processed_fa = process_posts_batch(batch_size=1000, source='fireant')
            processed = processed_fb + processed_yt + processed_fa
        
        if processed == 0:
            print("No posts to process.")
        else:
            print(f"Successfully processed {processed} posts.")
            
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