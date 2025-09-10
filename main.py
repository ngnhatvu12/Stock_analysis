import psycopg2
from config.database import get_db_connection
from services.text_summarizer import TextSummarizer
from services.stock_extractor import StockExtractor
from services.sentiment_analyzer import SentimentAnalyzer
from models.models import create_summary_table, create_reply_summary_table
import time
from datetime import datetime
import unicodedata

def process_posts_batch(batch_size=20):
    summarizer = TextSummarizer()
    stock_extractor = StockExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    
    create_summary_table()
    create_reply_summary_table()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT p.post_id, p.content 
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
        
        for post_id, content in posts:
            try:
                print(f"Processing post: {post_id}")
                
                if not content or len(content.strip()) < 10:
                    cur.execute("""
                        INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (post_id, "Content too short", None, None, "TRUNG_TINH"))
                    print(f"Skipped: {post_id} ")
                    continue
                
                content = unicodedata.normalize('NFC', content)
                summary = summarizer.summarize(content, max_length=100, min_length=20)
                
                stock_codes = stock_extractor.extract_stock_codes(content)
                
                cur.execute("DELETE FROM post_summary WHERE post_id = %s", (post_id,))
                
                if not stock_codes:
                    cur.execute("""
                        INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (post_id, summary, None, None, "TRUNG_TINH"))
                    print(f"Processed: {post_id} - No stock codes found")
                else:
                    # Trích xuất câu quan trọng cho mỗi mã chứng khoán
                    important_sentences = stock_extractor.extract_important_sentences(content, stock_codes)
                    
                    for stock_code in stock_codes:
                        important_sentence = important_sentences.get(stock_code, "")
                        
                        # Phân tích cảm xúc của câu quan trọng
                        sentiment = "TRUNG_TINH" 
                        if important_sentence:
                            sentiment_result = sentiment_analyzer.analyze_sentiment(important_sentence)
                            sentiment = sentiment_result["normalized_sentiment"]
                        
                        cur.execute("""
                            INSERT INTO post_summary 
                            (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (post_id, summary, stock_code, important_sentence, sentiment))
                        print(f"Processed: {post_id} - Stock: {stock_code} - Sentiment: {sentiment}")
                
                process_replies_for_post(post_id, summarizer, stock_extractor, sentiment_analyzer)
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    conn.commit()
                    print(f"Committed {processed_count} posts")
                
            except Exception as e:
                print(f"Error processing post {post_id}: {e}")
                cur.execute("""
                    INSERT INTO post_summary (post_id, content_tom_tat, ma_chung_khoan, cau_quan_trong, cam_xuc)
                    VALUES (%s, %s, %s, %s, %s)
                """, (post_id, f"Error: {str(e)[:100]}", None, None, "TRUNG_TINH"))
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

def process_replies_for_post(post_id, summarizer, stock_extractor, sentiment_analyzer):
    """Xử lý tất cả bình luận của một post"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT r.reply_id, r.rely_content 
            FROM fb_reply r 
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
        
        for reply_id, reply_content in replies:
            try:
                if not reply_content or len(reply_content.strip()) < 5:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (reply_id, post_id, "Content too short", None, None, "TRUNG_TINH"))
                    continue
                
                reply_content = unicodedata.normalize('NFC', reply_content)
                
                reply_summary = summarizer.summarize(reply_content, max_length=80, min_length=15)
                
                stock_codes = stock_extractor.extract_stock_codes(reply_content)
                
                cur.execute("DELETE FROM reply_summary WHERE reply_id = %s", (reply_id,))
                
                if not stock_codes:
                    cur.execute("""
                        INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (reply_id, post_id, reply_summary, None, None, "TRUNG_TINH"))
                    print(f"Processed reply: {reply_id} - No stock codes")
                else:
                    # Trích xuất câu quan trọng cho mỗi mã chứng khoán
                    important_sentences = stock_extractor.extract_important_sentences(reply_content, stock_codes)
                    
                    for stock_code in stock_codes:
                        important_sentence = important_sentences.get(stock_code, "")
                        
                        # Phân tích cảm xúc của câu quan trọng
                        sentiment = "TRUNG_TINH"  # Mặc định
                        if important_sentence:
                            sentiment_result = sentiment_analyzer.analyze_sentiment(important_sentence)
                            sentiment = sentiment_result["normalized_sentiment"]
                        
                        cur.execute("""
                            INSERT INTO reply_summary 
                            (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (reply_id, post_id, reply_summary, stock_code, important_sentence, sentiment))
                        print(f"Processed reply: {reply_id} - Stock: {stock_code} - Sentiment: {sentiment}")
                
                reply_count += 1
                
                if reply_count % 10 == 0:
                    conn.commit()
                    print(f"Committed {reply_count} replies for post {post_id}")
                
            except Exception as e:
                print(f"Error processing reply {reply_id}: {e}")
                cur.execute("""
                    INSERT INTO reply_summary (reply_id, post_id, rely_summary, stock_id, cau_quan_trong, cam_xuc)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (reply_id, post_id, f"Error: {str(e)[:100]}", None, None, "TRUNG_TINH"))
                continue
        
        conn.commit()
        print(f"Completed processing {reply_count} replies for post {post_id}")
        
    except Exception as e:
        print(f"Error processing replies for post {post_id}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def main():
    """Hàm main chạy liên tục"""
    batch_size = 10  
    
    while True:
        try:
            print(f"\n{datetime.now()} - Starting new batch processing...")
            processed = process_posts_batch(batch_size)
            
            if processed == 0:
                print("No more posts to process. Waiting 60 seconds...")
                time.sleep(60)
            else:
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            time.sleep(120)

if __name__ == "__main__":
    main()