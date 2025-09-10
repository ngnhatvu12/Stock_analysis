import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

def test_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        print("✅ Kết nối database thành công!")
        
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'fb_post'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            print("Bảng fb_post tồn tại")
        else:
            print("Bảng fb_post không tồn tại")
            
        cur.close()
        conn.close()
        
    except psycopg2.OperationalError as e:
        print(f"Lỗi kết nối: {e}")
        
    except Exception as e:
        print(f"Lỗi khác: {e}")

if __name__ == "__main__":
    test_connection()