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
                content_tom_tat TEXT,
                ma_chung_khoan VARCHAR(10),
                cau_quan_trong TEXT,
                confidence_score FLOAT DEFAULT 0.5,
                cam_xuc VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cur.execute(create_table_query)
            
            cur.execute("CREATE INDEX idx_post_summary_post_id ON post_summary(post_id)")
            cur.execute("CREATE INDEX idx_post_summary_ma_chung_khoan ON post_summary(ma_chung_khoan)")
            cur.execute("CREATE INDEX idx_post_summary_cam_xuc ON post_summary(cam_xuc)")
            
            conn.commit()
            print("post_summary table created successfully!")
        else:
            print("post_summary table already exists - columns updated")
            
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(reply_id, stock_id)
            )
            """
            cur.execute(create_table_query)
            
            cur.execute("CREATE INDEX idx_reply_summary_reply_id ON reply_summary(reply_id)")
            cur.execute("CREATE INDEX idx_reply_summary_post_id ON reply_summary(post_id)")
            cur.execute("CREATE INDEX idx_reply_summary_stock_id ON reply_summary(stock_id)")
            cur.execute("CREATE INDEX idx_reply_summary_cam_xuc ON reply_summary(cam_xuc)")
            
            conn.commit()
            print("reply_summary table created successfully!")
        else:
            print("reply_summary table already exists - columns updated")
            
    except psycopg2.errors.DuplicateTable:
        print("reply_summary table already exists - continuing")
        conn.rollback()
    except Exception as e:
        print(f"Error in create_reply_summary_table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()