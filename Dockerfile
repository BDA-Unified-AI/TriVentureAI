FROM python:3.11-slim

# Install git and git-lfs
RUN apt-get update && \
    apt-get install -y git git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add a non-root user for better security
RUN useradd -m -u 1000 user

# Switch to non-root user
USER user

# Ensure pip, setuptools, and wheel are up to date
RUN python -m pip install --upgrade pip setuptools wheel

# Set PATH to include user installs
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt file and install dependencies
COPY --chown=user ./requirements.txt /app/requirements.txt

# Install only necessary dependencies from the requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create Saved_Model directory and download model from Hugging Face
RUN mkdir -p /app/Saved_Model && \
    cd /app && \
    git lfs install && \
    git clone https://huggingface.co/spaces/darkbreakerk/triventure_ai && \
    cp -r triventure_ai/Model_API/Saved_Model/* Saved_Model/ && \
    rm -rf triventure_ai

# Copy the rest of the application code to the working directory
COPY --chown=user . /app

# Expose the port that your app will run on
EXPOSE 7880

# Run the Python script when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7880"]