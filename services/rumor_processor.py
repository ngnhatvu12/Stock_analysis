# services/rumor_processor.py
from services.text_rewriter import TextRewriter
from config.database import get_db_connection
import time
import psycopg2
import re

class RumorProcessor:
    def __init__(self):
        self.text_rewriter = TextRewriter()
    
    def process_rumors_batch(self, batch_size=50, last_24h_only=False):
        """
        Xử lý một batch dữ liệu từ bảng rumor và chuyển sang rumor_analyst
        Chỉ viết lại nội dung, giữ nguyên sentiment và symbol
        """
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Xây dựng query dựa trên thời gian
            if last_24h_only:
                current_timestamp = int(time.time())
                twenty_four_hours_ago = current_timestamp - (24 * 60 * 60)
                
                cur.execute("""
                    SELECT r.id, r.timestamp, r.symbol, r.content, r.sentiment, r.source
                    FROM rumor r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM rumor_analyst ra WHERE ra.processed_from_rumor_id = r.id
                    )
                    AND r.timestamp >= %s
                    ORDER BY r.timestamp DESC
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT r.id, r.timestamp, r.symbol, r.content, r.sentiment, r.source
                    FROM rumor r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM rumor_analyst ra WHERE ra.processed_from_rumor_id = r.id
                    )
                    ORDER BY r.timestamp DESC
                    LIMIT %s
                """, (batch_size,))
            
            rumors = cur.fetchall()
            
            if not rumors:
                time_constraint = " from last 24 hours" if last_24h_only else ""
                print(f"No rumors{time_constraint} to process.")
                return 0
            
            processed_count = 0
            error_count = 0
            
            for rumor_id, timestamp, symbol, content, sentiment, source in rumors:
                try:
                    print(f"\n--- Processing rumor {rumor_id} ---")
                    print(f"Original: {content}")
                    
                    # Kiểm tra dữ liệu đầu vào
                    if not content or len(str(content).strip()) < 3:
                        print(f"Skipping rumor {rumor_id}: content too short")
                        continue
                    
                    # Bước duy nhất: Viết lại nội dung
                    rewritten_content = self.text_rewriter.rewrite_rumor_text(str(content))
                    
                    # Kiểm tra kết quả viết lại
                    if not rewritten_content or len(rewritten_content.strip()) < 3:
                        print(f"Rewritten content too short for rumor {rumor_id}, using original")
                        rewritten_content = str(content)
                    
                    print(f"Rewritten: {rewritten_content}")
                    
                    # Xử lý symbol để tránh lỗi độ dài
                    safe_symbol = self._validate_symbol(symbol)
                    
                    # Chuẩn bị dữ liệu để insert
                    insert_data = (
                        str(content)[:1000],  # original_content (giới hạn độ dài)
                        str(rewritten_content)[:1000],  # rewritten_content (giới hạn độ dài)
                        safe_symbol,  # symbol (đã được validate)
                        sentiment,  # sentiment (có thể là None)
                        source,  # source (có thể là None)
                        timestamp,  # timestamp
                        rumor_id  # processed_from_rumor_id
                    )
                    
                    # Thực hiện insert
                    cur.execute("""
                        INSERT INTO rumor_analyst 
                        (original_content, rewritten_content, symbol, sentiment, source, timestamp, processed_from_rumor_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, insert_data)
                    
                    processed_count += 1
                    print(f"✓ Processed rumor {rumor_id}")
                    
                    # Commit thường xuyên hơn để tránh transaction dài
                    if processed_count % 5 == 0:
                        conn.commit()
                        print(f"Committed {processed_count} rumors")
                
                except psycopg2.Error as e:
                    error_count += 1
                    print(f"✗ Database error processing rumor {rumor_id}: {e}")
                    # Rollback transaction hiện tại và tiếp tục
                    conn.rollback()
                    continue
                    
                except Exception as e:
                    error_count += 1
                    print(f"✗ General error processing rumor {rumor_id}: {e}")
                    # Rollback transaction hiện tại và tiếp tục
                    conn.rollback()
                    continue
            
            # Commit các bản ghi còn lại
            if processed_count > 0:
                conn.commit()
                print(f"✅ Successfully processed {processed_count} rumors")
            
            if error_count > 0:
                print(f"⚠️  Encountered {error_count} errors during processing")
                
            return processed_count
            
        except Exception as e:
            print(f"❌ Error processing rumors batch: {e}")
            conn.rollback()
            return 0
        finally:
            cur.close()
            conn.close()
    
    def _validate_symbol(self, symbol):
        """Validate và làm sạch symbol để tránh lỗi database"""
        if not symbol:
            return None
        
        symbol_str = str(symbol).strip().upper()
        
        # Giới hạn độ dài symbol (thường mã chứng khoán VN <= 10 ký tự)
        if len(symbol_str) > 10:
            # Cắt bớt nếu quá dài, nhưng ưu tiên giữ phần đầu
            symbol_str = symbol_str[:10]
            print(f"Warning: Symbol truncated to {symbol_str}")
        
        # Loại bỏ ký tự không hợp lệ
        symbol_str = re.sub(r'[^A-Z0-9]', '', symbol_str)
        
        return symbol_str if symbol_str else None
    
    def process_rumors_safe(self, batch_size=50, last_24h_only=False):
        """
        Phiên bản an toàn hơn - xử lý từng bản ghi với transaction riêng
        """
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            # Xây dựng query dựa trên thời gian
            if last_24h_only:
                current_timestamp = int(time.time())
                twenty_four_hours_ago = current_timestamp - (24 * 60 * 60)
                
                cur.execute("""
                    SELECT r.id, r.timestamp, r.symbol, r.content, r.sentiment, r.source
                    FROM rumor r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM rumor_analyst ra WHERE ra.processed_from_rumor_id = r.id
                    )
                    AND r.timestamp >= %s
                    ORDER BY r.timestamp DESC
                    LIMIT %s
                """, (twenty_four_hours_ago, batch_size))
            else:
                cur.execute("""
                    SELECT r.id, r.timestamp, r.symbol, r.content, r.sentiment, r.source
                    FROM rumor r
                    WHERE NOT EXISTS (
                        SELECT 1 FROM rumor_analyst ra WHERE ra.processed_from_rumor_id = r.id
                    )
                    ORDER BY r.timestamp DESC
                    LIMIT %s
                """, (batch_size,))
            
            rumors = cur.fetchall()
            
            if not rumors:
                time_constraint = " from last 24 hours" if last_24h_only else ""
                print(f"No rumors{time_constraint} to process.")
                return 0
            
            processed_count = 0
            
            for rumor_id, timestamp, symbol, content, sentiment, source in rumors:
                # Tạo connection mới cho mỗi bản ghi để tránh lỗi transaction
                single_conn = get_db_connection()
                single_cur = single_conn.cursor()
                
                try:
                    print(f"\n--- Processing rumor {rumor_id} ---")
                    print(f"Original: {content}")
                    
                    # Kiểm tra dữ liệu đầu vào
                    if not content or len(str(content).strip()) < 3:
                        print(f"Skipping rumor {rumor_id}: content too short")
                        single_cur.close()
                        single_conn.close()
                        continue
                    
                    # Viết lại nội dung
                    rewritten_content = self.text_rewriter.rewrite_rumor_text(str(content))
                    
                    if not rewritten_content or len(rewritten_content.strip()) < 3:
                        print(f"Rewritten content too short for rumor {rumor_id}, using original")
                        rewritten_content = str(content)
                    
                    print(f"Rewritten: {rewritten_content}")
                    
                    # Xử lý symbol để tránh lỗi
                    safe_symbol = self._validate_symbol(symbol)
                    
                    # Insert với transaction riêng
                    single_cur.execute("""
                        INSERT INTO rumor_analyst 
                        (original_content, rewritten_content, symbol, sentiment, source, timestamp, processed_from_rumor_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(content)[:1000],
                        str(rewritten_content)[:1000],
                        safe_symbol,
                        sentiment,
                        source,
                        timestamp,
                        rumor_id
                    ))
                    
                    single_conn.commit()
                    processed_count += 1
                    print(f"✓ Processed rumor {rumor_id}")
                    
                except Exception as e:
                    print(f"✗ Error processing rumor {rumor_id}: {e}")
                    single_conn.rollback()
                    
                finally:
                    single_cur.close()
                    single_conn.close()
            
            print(f"✅ Successfully processed {processed_count} rumors")
            return processed_count
            
        except Exception as e:
            print(f"❌ Error in process_rumors_safe: {e}")
            return 0
        finally:
            cur.close()
            conn.close()
    
    def get_processing_stats(self):
        """Lấy thống kê về dữ liệu đã xử lý"""
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("SELECT COUNT(*) FROM rumor")
            total_rumors = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(DISTINCT processed_from_rumor_id) FROM rumor_analyst")
            processed_rumors = cur.fetchone()[0]
            
            return {
                'total_rumors': total_rumors,
                'processed_rumors': processed_rumors,
                'processing_rate': (processed_rumors / total_rumors * 100) if total_rumors > 0 else 0
            }
            
        except Exception as e:
            print(f"Error getting processing stats: {e}")
            return {}
        finally:
            cur.close()
            conn.close()
    
    def check_table_exists(self):
        """Kiểm tra xem bảng rumor_analyst có tồn tại không"""
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'rumor_analyst'
                )
            """)
            exists = cur.fetchone()[0]
            print(f"rumor_analyst table exists: {exists}")
            return exists
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False
        finally:
            cur.close()
            conn.close()