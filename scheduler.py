import schedule
import time
from datetime import datetime, time as dt_time
from main import process_posts_batch
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def daily_processing_job():
    """Công việc xử lý hàng ngày vào cuối ngày"""
    try:
        logger.info("Starting daily batch processing job...")
        
        processed_count = process_posts_batch(batch_size=1000)  
        
        logger.info(f"Daily processing completed. Processed {processed_count} posts.")
        
    except Exception as e:
        logger.error(f"Error in daily processing job: {e}")

def setup_scheduler():
    """Thiết lập lịch chạy hàng ngày"""
    schedule.every().day.at("23:50").do(daily_processing_job)  
    logger.info("Scheduler setup complete. Waiting for scheduled time...")
    logger.info("Next run scheduled for 23:50 daily")

def main():
    """Hàm main cho scheduler"""
    logger.info("Starting scheduler service...")
    
    setup_scheduler()
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in scheduler: {e}")
            time.sleep(300)  

if __name__ == "__main__":
    main()