FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/media/uploads /app/media/results
RUN chmod -R 777 /app/media

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "object_detection.wsgi:application"]