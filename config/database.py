import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            port=os.getenv('DB_PORT'),
            client_encoding='UTF8'
        )
        
        with conn.cursor() as cur:
            cur.execute("SET client_encoding TO 'UTF8'")
        
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

def test_encoding():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW client_encoding")
            encoding = cur.fetchone()[0]
            print(f"Database encoding: {encoding}")
    finally:
        conn.close()