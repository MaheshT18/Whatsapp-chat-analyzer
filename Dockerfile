# Step 1: Choose a slim base image
FROM python:3.12-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy only requirements first (better caching)
COPY requirements.txt .

# Step 4: Install system dependencies & Python packages in one layer
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader stopwords \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 5: Copy project files
COPY . .

# Step 6: Expose Streamlit port
EXPOSE 8501

# Step 7: Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
