FROM nvcr.io/nvidia/pytorch:24.07-py3

# Removed uv installation
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# ENV UV_COMPILE_BYTECODE=1 # Removed

# Install any system packages if needed by requirements (e.g., build essentials for some packages)
# RUN apt-get update && apt-get install -y ... && rm -rf /var/lib/apt/lists/*

# Copy certificate into image
# Adjust path relative to the Docker build context (llm-granite-ft/)
COPY /mlflow-cert/ca.pem /etc/mlflow/certs/ca.pem

# Set environment variable for MLflow client to find the cert
ENV MLFLOW_TRACKING_SERVER_CERT_PATH=/etc/mlflow/certs/ca.pem

# Set WORKDIR before copying files related to it
WORKDIR /workspace

RUN echo "CUDA Version in Base Image:" && nvcc --version | grep "release" && echo "GCC Version in Base Image:" && gcc --version

# ----------  Python deps via pip  ----------
COPY requirements.txt .
# COPY pyproject.toml . # Keep if other tools might use it, or remove if only for uv

RUN pip install --no-cache-dir -r requirements.txt

# ----------  Copy rest of the code  ----------
# Copy the rest of the project code into WORKDIR (/workspace)
COPY . .

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
# The CMD is usually overridden by srun, but good to have a default.
CMD ["python", "finetune.py"]