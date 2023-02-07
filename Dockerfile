# 基於 Python 的鏡像
FROM python:3.7-slim

# 設置工作目錄
WORKDIR /app/src

# 將專案文件夾拷貝到 Docker 鏡像中
COPY . /app

# 安裝專案需要的第三方庫
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y libgomp1 libquadmath0


# 執行專案
CMD ["python", "main.py"]