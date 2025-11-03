FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
pip install torchvision --index-url https://download.pytorch.org/whl/cpu && \
pip install -r requirements.txt

COPY . .

CMD ["python", "telegram_poetto_bot.py"]
