#!/bin/bash

set -e

# Parse arguments
if [ "$1" != "--run" ]; then
    echo "Usage: $0 --run <filename> [ccc arguments...]"
    echo ""
    echo "Example:"
    echo "  $0 --run hello.c -o hello"
    echo "  $0 --run test.c"
    exit 1
fi

shift  # remove --run

if [ -z "$1" ]; then
    echo "Error: filename required"
    echo "Usage: $0 --run <filename> [ccc arguments...]"
    exit 1
fi

filename="$1"
shift  # remove filename

# Check if file exists
if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' not found"
    exit 1
fi

# Get absolute path
abs_filename=$(cd "$(dirname "$filename")" && pwd)/$(basename "$filename")
work_dir=$(dirname "$abs_filename")
file_basename=$(basename "$filename")

# Determine output filename
output_name=""
remaining_args=()
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "-o" ]; then
        output_name="$arg"
    else
        remaining_args+=("$arg")
    fi
    prev_arg="$arg"
done

# If no -o flag, use filename without extension
if [ -z "$output_name" ]; then
    output_name="${file_basename%.c}"
fi

# Get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image if it doesn't exist
image_name="ccc-compiler"
if ! docker image inspect "$image_name" > /dev/null 2>&1; then
    echo "Building Docker image: $image_name"
    docker build -t "$image_name" "$script_dir"
fi

# Build extra args string (shell-escaped)
extra_args=""
for arg in "${remaining_args[@]}"; do
    extra_args+=" $(printf '%q' "$arg")"
done

# Compile the file inside the container with dynamic include paths
echo "Compiling $filename with ccc in Docker..."
docker run --rm \
    -v "$work_dir:/workspace" \
    --entrypoint /bin/sh \
    -e OUTPUT_NAME="$output_name" \
    -e FILE_BASENAME="$file_basename" \
    -e EXTRA_ARGS="$extra_args" \
    "$image_name" \
    -lc '
set -e
ARCH=$(gcc -dumpmachine)
INCDIR=$(gcc -print-file-name=include)
if echo "$ARCH" | grep -q "aarch64"; then
    CCC=/app/target/release/ccc-arm
else
    CCC=/app/target/release/ccc
fi

"$CCC" \
    -isystem /usr/include \
    -isystem "/usr/include/${ARCH}" \
    -isystem "${INCDIR}" \
    -L "/usr/lib/${ARCH}" \
    -L "/lib/${ARCH}" \
    -lc \
    -o "/workspace/${OUTPUT_NAME}" \
    "/workspace/${FILE_BASENAME}" ${EXTRA_ARGS}
'

echo ""
echo "Running compiled binary..."
docker run --rm \
    -v "$work_dir:/workspace" \
    --entrypoint /bin/sh \
    -e OUTPUT_NAME="$output_name" \
    "$image_name" \
    -lc '
ARCH=$(gcc -dumpmachine)
export LD_LIBRARY_PATH="/lib/${ARCH}:/usr/lib/${ARCH}:${LD_LIBRARY_PATH}"
exec "/workspace/${OUTPUT_NAME}"
'
