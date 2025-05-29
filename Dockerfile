# Use the official PyTorch image with CUDA (or CPU-only if you prefer)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy your Python code into the container
COPY . /app

# Install any additional packages (like argparse and numpy are preinstalled)
RUN pip install --upgrade pip && \
    pip install numpy

# Optional: if you use any private modules like `packages.model`, copy them
# COPY packages/ /app/packages/

# Set default command
CMD ["python", "train_pangolin.py"]

