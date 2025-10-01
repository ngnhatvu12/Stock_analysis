FROM python:3.9-slim

WORKDIR /app

# Copy requirements trước để tận dụng cache Docker
COPY requirements.txt .

# Cài đặt dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Tạo thư mục log
RUN mkdir -p logs

# Expose port nếu cần
# EXPOSE 8000

# Chạy scheduler thay vì main.py
CMD ["python", "scheduler.py"]