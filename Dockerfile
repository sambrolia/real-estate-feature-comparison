FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY data/ data/

# Expose the port 3000 for the Flask app
EXPOSE 3000

CMD ["python", "src/app.py"]
