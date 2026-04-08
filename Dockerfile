FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY app app
COPY src src
COPY templates templates
COPY static static
COPY models models

EXPOSE 8000

CMD ["python", "-m", "flask", "--app", "app/app.py", "run", "--host=0.0.0.0", "--port=8000"]
