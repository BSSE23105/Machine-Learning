FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY flask_app.py .
COPY model.joblib .
COPY templates/ templates/

EXPOSE 5000

CMD ["python", "flask_app.py"]
