import psycopg2
from config.database import get_db_connection

def create_summary_table():
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'post_summary'
            )
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            create_table_query = """
            CREATE TABLE post_summary (
                id SERIAL PRIMARY KEY,
                post_id VARCHAR(50) NOT NULL,
                ma_chung_khoan VARCHAR(10),
                cau_quan_trong TEXT,
                confidence_score FLOAT DEFAULT 0.5,
                cam_xuc VARCHAR(20),
                timestamp BIGINT,
                source VARCHAR(20) DEFAULT 'facebook'
            )
            """
            cur.execute(create_table_query)
            
            cur.execute("CREATE INDEX idx_post_summary_post_id ON post_summary(post_id)")
            cur.execute("CREATE INDEX idx_post_summary_ma_chung_khoan ON post_summary(ma_chung_khoan)")
            cur.execute("CREATE INDEX idx_post_summary_cam_xuc ON post_summary(cam_xuc)")
            cur.execute("CREATE INDEX idx_post_summary_timestamp ON post_summary(timestamp)")
            cur.execute("CREATE INDEX idx_post_summary_source ON post_summary(source)")
            
            conn.commit()
            print("post_summary table created successfully!")
        else:
            try:
                # Thêm cột source nếu chưa có
                cur.execute("ALTER TABLE post_summary ADD COLUMN source VARCHAR(20) DEFAULT 'facebook'")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_post_summary_source ON post_summary(source)")
                conn.commit()
                print("Added source column to post_summary table")
            except psycopg2.errors.DuplicateColumn:
                print("source column already exists")
                conn.rollback()
            
    except psycopg2.errors.DuplicateTable:
        print("post_summary table already exists - continuing")
        conn.rollback()
    except Exception as e:
        print(f"Error in create_summary_table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def create_reply_summary_table():
    """Tạo bảng reply_summary nếu chưa tồn tại"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'reply_summary'
            )
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            create_table_query = """
            CREATE TABLE reply_summary (
                id SERIAL PRIMARY KEY,
                reply_id VARCHAR(50) NOT NULL,
                post_id VARCHAR(50) NOT NULL,
                rely_summary TEXT,
                stock_id VARCHAR(10),
                cau_quan_trong TEXT,
                confidence_score FLOAT DEFAULT 0.5,
                cam_xuc VARCHAR(20),
                timestamp BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(20) DEFAULT 'facebook',
                UNIQUE(reply_id, stock_id)
            )
            """
            cur.execute(create_table_query)
            
            cur.execute("CREATE INDEX idx_reply_summary_reply_id ON reply_summary(reply_id)")
            cur.execute("CREATE INDEX idx_reply_summary_post_id ON reply_summary(post_id)")
            cur.execute("CREATE INDEX idx_reply_summary_stock_id ON reply_summary(stock_id)")
            cur.execute("CREATE INDEX idx_reply_summary_cam_xuc ON reply_summary(cam_xuc)")
            cur.execute("CREATE INDEX idx_reply_summary_timestamp ON reply_summary(timestamp)")
            cur.execute("CREATE INDEX idx_reply_summary_source ON reply_summary(source)")
            
            conn.commit()
            print("reply_summary table created successfully!")
        else:
            try:
                cur.execute("ALTER TABLE reply_summary ADD COLUMN source VARCHAR(20) DEFAULT 'facebook'")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_reply_summary_source ON reply_summary(source)")
                conn.commit()
                print("Added source column to reply_summary table")
            except psycopg2.errors.DuplicateColumn:
                print("source column already exists")
                conn.rollback()
            
    except psycopg2.errors.DuplicateTable:
        print("reply_summary table already exists - continuing")
        conn.rollback()
    except Exception as e:
        print(f"Error in create_reply_summary_table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
        
def create_news_statistics_table():
    """Tạo bảng news_statistics nếu chưa tồn tại"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'news_statistics'
            )
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            create_table_query = """
            CREATE TABLE news_statistics (
                id SERIAL PRIMARY KEY,
                timestamp BIGINT NOT NULL,
                total INTEGER NOT NULL,
                total_positive INTEGER NOT NULL,
                total_negative INTEGER NOT NULL,
                total_neutral INTEGER NOT NULL,
                news INTEGER NOT NULL,
                fireant INTEGER NOT NULL,
                zalo INTEGER NOT NULL,
                facebook INTEGER NOT NULL,
                youtube INTEGER NOT NULL,
                tiktok INTEGER NOT NULL,
                aim_score DOUBLE PRECISION
            )
            """
            cur.execute(create_table_query)
            
            cur.execute("CREATE INDEX idx_news_statistics_timestamp ON news_statistics(timestamp)")
            
            conn.commit()
            print("news_statistics table created successfully!")
        else:
            print("news_statistics table already exists")
            
    except Exception as e:
        print(f"Error in create_news_statistics_table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
        
def create_rumor_analyst_table():
    """Tạo bảng rumor_analyst nếu chưa tồn tại"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'rumor_analyst'
            )
        """)
        
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            create_table_query = """
            CREATE TABLE rumor_analyst (
                id SERIAL PRIMARY KEY,
                original_content TEXT NOT NULL,
                rewritten_content TEXT NOT NULL,
                symbol VARCHAR(20),
                sentiment VARCHAR(20),
                source VARCHAR(50),
                timestamp BIGINT,
                processed_from_rumor_id INTEGER,
                FOREIGN KEY (processed_from_rumor_id) REFERENCES rumor(id)
            )
            """
            cur.execute(create_table_query)
            
            # Tạo indexes
            cur.execute("CREATE INDEX idx_rumor_analyst_symbol ON rumor_analyst(symbol)")
            cur.execute("CREATE INDEX idx_rumor_analyst_sentiment ON rumor_analyst(sentiment)")
            cur.execute("CREATE INDEX idx_rumor_analyst_timestamp ON rumor_analyst(timestamp)")
            cur.execute("CREATE INDEX idx_rumor_analyst_source ON rumor_analyst(source)")
            cur.execute("CREATE INDEX idx_rumor_analyst_rumor_id ON rumor_analyst(processed_from_rumor_id)")
            
            conn.commit()
            print("rumor_analyst table created successfully!")
        else:
            print("rumor_analyst table already exists")
            
    except Exception as e:
        print(f"Error in create_rumor_analyst_table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()