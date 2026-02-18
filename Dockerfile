FROM rust:latest

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    gcc-aarch64-linux-gnu \
    gcc-riscv64-linux-gnu \
    gcc-arm-linux-gnueabihf \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the source code
COPY . .

# Build the ccc compiler in release mode
RUN cargo build --release

# Create a working directory for files to compile
WORKDIR /workspace

# Set the default binary to ccc (x86-64)
ENTRYPOINT ["/app/target/release/ccc"]

# Default command: show usage
CMD ["--help"]
