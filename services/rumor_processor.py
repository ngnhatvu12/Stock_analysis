from services.text_rewriter_enhanced import TextRewriter
from config.database import get_db_connection
import time
import psycopg2
import re
import os

class RumorProcessor:
    def __init__(self, use_trained_model=False):
        """
        Khởi tạo RumorProcessor
        
        Args:
            use_trained_model (bool): Nếu True, sử dụng model đã huấn luyện
        """
        self.use_trained_model = use_trained_model
        self.text_rewriter = None
        self._initialize_rewriter()
        
    def _initialize_rewriter(self):
        """Khởi tạo text rewriter với model phù hợp"""
        try:
            if self.use_trained_model:
                print("🔄 Initializing with TRAINED rewriter model...")
                # Sử dụng model đã huấn luyện
                trained_model_path = "./trained_rewriter_model"
                
                if os.path.exists(trained_model_path):
                    self.text_rewriter = TextRewriter(trained_model_path)
                    print("✅ Trained rewriter model loaded successfully!")
                else:
                    print("⚠️ Trained model not found, using base model instead")
                    self.text_rewriter = TextRewriter()
            else:
                print("🔄 Initializing with BASE rewriter model...")
                # Sử dụng model gốc
                self.text_rewriter = TextRewriter()
                
        except Exception as e:
            print(f"❌ Error initializing text rewriter: {e}")
            # Fallback to base model
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
                model_type = "TRAINED" if self.use_trained_model else "BASE"
                print(f"No rumors{time_constraint} to process with {model_type} model.")
                return 0
            
            processed_count = 0
            error_count = 0
            
            for rumor_id, timestamp, symbol, content, sentiment, source in rumors:
                try:
                    model_type = "TRAINED" if self.use_trained_model else "BASE"
                    print(f"\n--- Processing rumor {rumor_id} ({model_type} model) ---")
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
                model_type = "TRAINED" if self.use_trained_model else "BASE"
                print(f"✅ Successfully processed {processed_count} rumors with {model_type} model")
            
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