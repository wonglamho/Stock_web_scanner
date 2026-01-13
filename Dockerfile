FROM python:3.10-slim

ENV TZ=Asia/Shanghai
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY app.py .

# 暴露端口
EXPOSE 8501

# 启动
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
