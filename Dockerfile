FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends fonts-dejavu-core
RUN rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime code, Streamlit config, and the labeled demo dataset.
COPY app.py wastelens_aggregate.py wastelens_config.py wastelens_core.py wastelens_observability.py wastelens_store.py wastelens_theme.py ./
COPY .streamlit/config.toml .streamlit/config.toml
COPY data/synthetic_historical_data.csv data/synthetic_historical_data.csv

# Expose Streamlit port
EXPOSE 8501

ENV WASTELENS_ENV=production
ENV WASTELENS_DB_PATH=/app/.wastelens/wastelens.sqlite3

# Mount a persistent volume here to retain user data outside the container layer.
VOLUME ["/app/.wastelens"]

# Command to run the application
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
