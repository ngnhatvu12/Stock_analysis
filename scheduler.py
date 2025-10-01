import schedule
import time
import subprocess
import sys
import os
from datetime import datetime
import logging

# Thêm thư mục gốc vào path để import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    """Công việc xử lý hàng ngày vào cuối ngày với --last-24h"""
    try:
        logger.info("Starting daily batch processing with --last-24h...")
        
        # Chạy main.py với tham số --last-24h
        result = subprocess.run([
            sys.executable, "main.py", "--last-24h"
        ], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Daily processing completed successfully!")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"Daily processing failed with return code: {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error in daily processing job: {e}")

def health_check_job():
    """Kiểm tra sức khỏe hệ thống hàng giờ"""
    try:
        logger.info("Performing system health check...")
        
        # Kiểm tra kết nối database và các service
        result = subprocess.run([
            sys.executable, "main.py", "--stats", "1"
        ], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("System health check passed")
        else:
            logger.warning("System health check warning")
            
    except Exception as e:
        logger.error(f"Health check error: {e}")

def setup_scheduler():
    """Thiết lập lịch chạy hàng ngày"""
    # Chạy chính vào 23:50 hàng ngày
    schedule.every().day.at("23:50").do(daily_processing_job)
    
    # Health check hàng giờ từ 8h đến 23h
    for hour in range(8, 24):
        schedule.every().day.at(f"{hour:02d}:00").do(health_check_job)
    
    logger.info("Scheduler setup complete!")
    logger.info("Next run scheduled for 23:50 daily")
    logger.info("Health checks scheduled hourly from 8:00 to 23:00")

def print_next_runs():
    """In thông tin về lần chạy tiếp theo"""
    try:
        from schedule import next_run
        next_run_time = next_run(schedule.jobs[0])
        logger.info(f"Next scheduled run: {next_run_time}")
    except:
        pass

def main():
    """Hàm main cho scheduler"""
    logger.info("Starting AI Sentiment Analysis Scheduler Service...")
    
    setup_scheduler()
    print_next_runs()
    
    logger.info("Scheduler is running. Press Ctrl+C to stop.")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Kiểm tra mỗi 30 giây
            
            # Log mỗi 10 phút để biết service vẫn chạy
            if int(time.time()) % 600 == 0:
                logger.info("Scheduler service is alive and waiting...")
                
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in scheduler: {e}")
            time.sleep(60)  # Chờ 1 phút nếu có lỗi

if __name__ == "__main__":
    main()