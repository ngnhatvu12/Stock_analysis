import psycopg2
from config.database import get_db_connection
from services.stock_extractor import StockExtractor
from services.sentiment_analyzer import SentimentAnalyzer
from models.models import create_all_content_table, create_reply_summary_table, create_news_statistics_table, create_rumor_analyst_table
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
    
    create_all_content_table()
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
                        SELECT 1 FROM all_content ac WHERE ac.content = p.content AND ac.source = 'facebook'
                    )
                    AND p.timestamp >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT p.post_id, p.content, p.timestamp 
                    FROM fb_post p 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_content ac WHERE ac.content = p.content AND ac.source = 'facebook'
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
                        SELECT 1 FROM all_content ac WHERE ac.content = y.post_sentence AND ac.source = 'youtube'
                    )
                    AND EXTRACT(epoch FROM y.post_at) >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT y.post_id, y.post_sentence, EXTRACT(epoch FROM y.post_at)::bigint
                    FROM yt_post_summary y 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_content ac WHERE ac.content = y.post_sentence AND ac.source = 'youtube'
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
                        SELECT 1 FROM all_content ac WHERE ac.content = f.original_content AND ac.source = 'fireant'
                    )
                    AND EXTRACT(epoch FROM f.date) >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT f.post_id, f.original_content, EXTRACT(epoch FROM f.date)::bigint
                    FROM fireant_posts f 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_content ac WHERE ac.content = f.original_content AND ac.source = 'fireant'
                    )
                    LIMIT %s
                """, (batch_size,))
        elif source == 'zalo':
            # Xử lý dữ liệu từ Zalo
            if last_24h_only:
                cur.execute("""
                    SELECT z.id, z.content, z.date
                    FROM zalo_chat z 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_content ac WHERE ac.content = z.content AND ac.source = 'zalo'
                    )
                    AND z.date >= %s
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT z.id, z.content, z.date
                    FROM zalo_chat z 
                    WHERE NOT EXISTS (
                        SELECT 1 FROM all_content ac WHERE ac.content = z.content AND ac.source = 'zalo'
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
                        INSERT INTO all_content (timestamp, source, title, symbol, content, sentiment)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (post_timestamp, source, None, None, content, "TRUNG_TÍNH"))
                    print(f"Skipped: {post_id_str} ")
                    continue
                
                content = unicodedata.normalize('NFC', content)
                
                stock_codes = stock_extractor.extract_stock_codes(content)
                
                # Không cần DELETE vì đã có constraint UNIQUE
                
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
                    
                    cur.execute("""
                    INSERT INTO all_content (timestamp, source, title, symbol, content, sentiment)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (post_timestamp, source, None, None, important_sentence, sentiment))
                  
                    print(f"Processed: {post_id_str} - No stock codes - Sentiment: {sentiment}")
                else:
                    # Trích xuất câu quan trọng cho mỗi mã chứng khoán
                    important_sentences = stock_extractor.extract_important_sentences(content, stock_codes)
                    
                    for stock_code in stock_codes:
                        important_sentence = important_sentences.get(stock_code, "")
                        
                        sentiment = "TRUNG_TÍNH" 
                        
                        if important_sentence:
                            # Sử dụng custom model nếu available
                            if hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                                sentiment_result = sentiment_analyzer.analyze_with_custom_model(important_sentence)
                            else:
                                sentiment_result = sentiment_analyzer.analyze_sentiment(important_sentence)
                            sentiment = sentiment_result["normalized_sentiment"]
                        else:
                            # Sử dụng custom model nếu available
                            if hasattr(sentiment_analyzer, 'analyze_with_custom_model'):
                                sentiment_result = sentiment_analyzer.analyze_with_custom_model(content)
                            else:
                                sentiment_result = sentiment_analyzer.analyze_sentiment(content)
                            sentiment = sentiment_result["normalized_sentiment"]
                            
                        cur.execute("""
                        INSERT INTO all_content 
                        (timestamp, source, title, symbol, content, sentiment)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """, (post_timestamp, source, None, stock_code, important_sentence, sentiment))
                    
                    print(f"Processed: {post_id_str} - Stock: {stock_codes} - Multiple entries created")
                
                # Chỉ xử lý replies cho Facebook posts
                if source == 'facebook':
                    process_replies_for_post(post_id_str, stock_extractor, sentiment_analyzer, conn, use_custom_model)
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    conn.commit()
                    print(f"Committed {processed_count} posts")
                
            except psycopg2.errors.UniqueViolation:
                print(f"Duplicate content for post {post_id_str} - skipping")
                conn.rollback()
                continue
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                try:
                    cur.execute("""
                        INSERT INTO all_content (timestamp, source, title, symbol, content, sentiment)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (post_timestamp, source, None, None, content, "TRUNG_TÍNH"))
                except psycopg2.errors.UniqueViolation:
                    print(f"Duplicate content for post {post_id_str} - skipping")
                    conn.rollback()
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

def process_rumors_batch(batch_size=100, last_24h_only=False, use_trained_model=False):
    """
    Xử lý dữ liệu từ bảng rumor và chuyển sang rumor_analyst
    Chỉ viết lại nội dung, giữ nguyên các thông tin khác
    """
    from services.rumor_processor import RumorProcessor
    
    model_type = "TRAINED" if use_trained_model else "BASE"
    print(f"Starting rumor processing with {model_type} model (batch_size={batch_size}, last_24h={last_24h_only})...")
    
    rumor_processor = RumorProcessor(use_trained_model=use_trained_model)
    processed_count = rumor_processor.process_rumors_batch(
        batch_size=batch_size, 
        last_24h_only=last_24h_only
    )
    
    if processed_count > 0:
        stats = rumor_processor.get_processing_stats()
        print(f"Rumor processing completed: {processed_count} rumors processed")
        print(f"Overall progress: {stats.get('processing_rate', 0):.1f}%")
    
    return processed_count

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
    """Huấn luyện lại mô hình cảm xúc với dữ liệu synthetic"""
    try:
        from training.data_preparation import TrainingDataPreparer
        from training.train_sentiment_model import train_comprehensive_model
        
        print("Starting comprehensive model retraining...")

        # SỬA: Đường dẫn đúng đến file synthetic data
        synthetic_data_path = "training_data/synthetic_stock_sentiment_20000.jsonl"
        
        # Load dữ liệu từ synthetic file
        preparer = TrainingDataPreparer(synthetic_data_path)
        preparer.create_training_dataset("training_data_complete.jsonl")
        
        # Huấn luyện mô hình
        success = train_comprehensive_model()
        
        if success:
            print("Comprehensive model retraining completed successfully!")
        else:
            print("Model retraining failed (not enough data)")
        return success
    except Exception as e:
        print(f"Model retraining failed: {e}")
        import traceback; traceback.print_exc()
        return False

def retrain_rewriter_model():
    """Huấn luyện mô hình viết lại rumor"""
    try:
        print("Starting rumor rewriter training...")
        from training.train_rewriter_model import train_rumor_rewriter_model
        success = train_rumor_rewriter_model()
        if success:
            print("✅ Rumor rewriter training completed successfully!")
        else:
            print("❌ Rumor rewriter training failed")
        return success
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def evaluate_rewriter_model():
    """Đánh giá mô hình viết lại rumor"""
    try:
        print("🔍 Evaluating rewriter model...")
        from training.evaluate_rewriter import evaluate_rewriter_model as eval_func
        results = eval_func()
        if results:
            print("✅ Rewriter model evaluation completed!")
        else:
            print("❌ Rewriter model evaluation failed")
        return results
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def evaluate_model():
    """Đánh giá mô hình đã huấn luyện"""
    try:
        from training.evaluate_model import evaluate_model as eval_func
        
        test_file = 'training_data_complete.jsonl' 
        if not os.path.exists(test_file):
            print("No training data found. Please run retraining first.")
            return False
            
        print("Evaluating trained model...")
        results = eval_func(test_file, './trained_sentiment_model')
        print("Evaluation completed!")
        return True
        
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return False

def calculate_daily_statistics():
    """Tính toán thống kê hàng ngày từ all_content"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Lấy timestamp của 0h hôm nay và 0h hôm qua
        from datetime import datetime, time
        today = datetime.now().date()
        today_start = int(datetime.combine(today, time(0, 0, 0)).timestamp())
        yesterday_start = today_start - (24 * 60 * 60)
        
        # Kiểm tra xem đã có thống kê cho ngày hôm nay chưa
        cur.execute("SELECT id FROM news_statistics WHERE timestamp = %s", (today_start,))
        existing_stat = cur.fetchone()
        
        if existing_stat:
            print(f"Statistics for {today} already exist. Skipping...")
            return
        
        # Thống kê tổng quan từ tất cả nguồn
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN sentiment = 'TÍCH_CỰC' THEN 1 END) as total_positive,
                COUNT(CASE WHEN sentiment = 'TIÊU_CỰC' THEN 1 END) as total_negative,
                COUNT(CASE WHEN sentiment = 'TRUNG_TÍNH' THEN 1 END) as total_neutral
            FROM all_content 
            WHERE timestamp >= %s AND timestamp < %s
        """, (yesterday_start, today_start))
        
        total_stats = cur.fetchone()
        
        # Thống kê theo từng nguồn
        sources = ['facebook', 'youtube', 'fireant', 'zalo', 'tiktok', 'news']
        source_counts = {}
        
        for source in sources:
            cur.execute("""
                SELECT COUNT(*) 
                FROM all_content 
                WHERE source = %s AND timestamp >= %s AND timestamp < %s
            """, (source, yesterday_start, today_start))
            source_counts[source] = cur.fetchone()[0] or 0
        
        # Tính aim_score (tỷ lệ tích cực)
        total_posts = total_stats[0] or 1
        positive_posts = total_stats[1] or 0
        aim_score = (positive_posts / total_posts) * 100
        
        # Chèn dữ liệu thống kê vào bảng
        cur.execute("""
            INSERT INTO news_statistics 
            (timestamp, total, total_positive, total_negative, total_neutral, 
             news, fireant, zalo, facebook, youtube, tiktok, aim_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            yesterday_start,  # Thống kê cho ngày hôm qua
            total_stats[0] or 0,
            total_stats[1] or 0,
            total_stats[2] or 0,
            total_stats[3] or 0,
            source_counts['news'],
            source_counts['fireant'],
            source_counts['zalo'],
            source_counts['facebook'],
            source_counts['youtube'],
            source_counts['tiktok'],
            aim_score
        ))
        
        conn.commit()
        
        print(f"Daily statistics calculated for {datetime.fromtimestamp(yesterday_start).date()}")
        print(f"Total: {total_stats[0]}, Positive: {total_stats[1]}, Negative: {total_stats[2]}, Neutral: {total_stats[3]}")
        
    except Exception as e:
        print(f"Error calculating daily statistics: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def get_recent_statistics(days=7):
    """Lấy thống kê của N ngày gần nhất"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Tính timestamp của N ngày trước
        from datetime import datetime, time, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        start_timestamp = int(datetime.combine(start_date, time(0, 0, 0)).timestamp())
        
        cur.execute("""
            SELECT 
                timestamp,
                total,
                total_positive,
                total_negative,
                total_neutral,
                news,
                fireant,
                zalo,
                facebook,
                youtube,
                tiktok,
                aim_score
            FROM news_statistics 
            WHERE timestamp >= %s 
            ORDER BY timestamp DESC
        """, (start_timestamp,))
        
        stats = cur.fetchall()
        
        # Format kết quả
        result = []
        for stat in stats:
            result.append({
                'date': datetime.fromtimestamp(stat[0]).strftime('%Y-%m-%d'),
                'total': stat[1],
                'total_positive': stat[2],
                'total_negative': stat[3],
                'total_neutral': stat[4],
                'news': stat[5],
                'fireant': stat[6],
                'zalo': stat[7],
                'facebook': stat[8],
                'youtube': stat[9],
                'tiktok': stat[10],
                'aim_score': stat[11]
            })
        
        return result
        
    except Exception as e:
        print(f"Error getting recent statistics: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def main():
    """Hàm main chạy một lần"""
    try:
        create_all_content_table()
        create_reply_summary_table()
        create_news_statistics_table()
        create_rumor_analyst_table()
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
            elif sys.argv[1] == "--retrain-rewriter":
                success = retrain_rewriter_model()
                return 0 if success else 1             
            elif sys.argv[1] == "--evaluate-rewriter":
                results = evaluate_rewriter_model()
                return 0 if results else 1
            elif sys.argv[1] == "--use-trained-rewriter":
                print("🔄 Using trained rewriter model...")
                # Xác định các tham số
                last_24h = "--last-24h" in sys.argv
                batch_size = 100
                
                # Xác định batch size nếu có tham số
                for arg in sys.argv:
                    if arg.startswith("--batch-size="):
                        try:
                            batch_size = int(arg.split("=")[1])
                        except:
                            pass
                
                processed = process_rumors_batch(
                    batch_size=batch_size,
                    last_24h_only=last_24h,
                    use_trained_model=True
                )
                print(f"✅ Processed {processed} rumors with TRAINED model")
                return 0
                
            elif sys.argv[1] == "--process-rumors":
                print("🔄 Processing rumors with base model...")
                last_24h = "--last-24h" in sys.argv
                batch_size = 100
                
                # Xác định batch size nếu có tham số
                for arg in sys.argv:
                    if arg.startswith("--batch-size="):
                        try:
                            batch_size = int(arg.split("=")[1])
                        except:
                            pass
                
                processed = process_rumors_batch(
                    batch_size=batch_size,
                    last_24h_only=last_24h,
                    use_trained_model=False
                )
                print(f"✅ Processed {processed} rumors with BASE model")
                return 0
            elif sys.argv[1] == "--export-excel":
                print("Exporting training data to Excel...")
                from training.data_preparation import TrainingDataPreparer             
                preparer = TrainingDataPreparer()             
                jsonl_file = "training_data_complete.jsonl"
                excel_file = "training_data.xlsx"
                if len(sys.argv) > 2:
                    excel_file = sys.argv[2]
                    if not excel_file.endswith('.xlsx'):
                        excel_file += '.xlsx'              
                if not os.path.exists(jsonl_file):
                    print(f"File {jsonl_file} không tồn tại. Đang tạo dataset mới...")
                    preparer.create_training_dataset(jsonl_file)
                success = preparer.export_to_excel(jsonl_file, excel_file)
                return 0 if success else 1  
            elif sys.argv[1] == "--use-custom":
                print(f"\n{datetime.now()} - Starting batch processing with custom model...")
                processed_fb = process_posts_batch(batch_size=1000, use_custom_model=True, source='facebook')
                processed_yt = process_posts_batch(batch_size=1000, use_custom_model=True, source='youtube')
                processed_fa = process_posts_batch(batch_size=1000, use_custom_model=True, source='fireant')
                processed_zalo = process_posts_batch(batch_size=1000, use_custom_model=True, source='zalo')
                processed = processed_fb + processed_yt + processed_fa + processed_zalo
            elif sys.argv[1] == "--youtube-only":
                print(f"\n{datetime.now()} - Processing YouTube posts only...")
                processed = process_posts_batch(batch_size=1000, source='youtube')
            elif sys.argv[1] == "--fireant-only":
                print(f"\n{datetime.now()} - Processing FireAnt posts only...")
                processed = process_posts_batch(batch_size=1000, source='fireant')
            elif sys.argv[1] == "--zalo-only":
                print(f"\n{datetime.now()} - Processing Zalo posts only...")
                processed = process_posts_batch(batch_size=1000, source='zalo')
            elif sys.argv[1] == "--last-24h":
                print(f"\n{datetime.now()} - Processing posts from last 24 hours with custom model...")
                processed_fb = process_posts_batch(batch_size=1000, use_custom_model=True, source='facebook', last_24h_only=True)
                processed_yt = process_posts_batch(batch_size=1000, use_custom_model=True, source='youtube', last_24h_only=True)
                processed_fa = process_posts_batch(batch_size=1000, use_custom_model=True, source='fireant', last_24h_only=True)
                processed_zalo = process_posts_batch(batch_size=1000, use_custom_model=True, source='zalo', last_24h_only=True)
                processed = processed_fb + processed_yt + processed_fa + processed_zalo
            else:
                print(f"Unknown argument: {sys.argv[1]}")
                print("Usage: python main.py [--retrain | --evaluate | --use-custom | --youtube-only | --fireant-only | --zalo-only | --last-24h | --stats [days]]")
                return 1
        else:
            print(f"\n{datetime.now()} - Starting batch processing with default model...")
            processed_fb = process_posts_batch(batch_size=1000, source='facebook')
            processed_yt = process_posts_batch(batch_size=1000, source='youtube')
            processed_fa = process_posts_batch(batch_size=1000, source='fireant')
            processed_zalo = process_posts_batch(batch_size=1000, source='zalo')
            processed = processed_fb + processed_yt + processed_fa + processed_zalo
        
        if processed == 0:
            print("No posts to process.")
        else:
            print(f"Successfully processed {processed} posts.")
        if processed > 0:
            print("Calculating daily statistics...")
            calculate_daily_statistics()
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    os.makedirs('training', exist_ok=True)
    main()