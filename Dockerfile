# syntax=docker/dockerfile:1

FROM rust:1.84-bookworm AS builder

WORKDIR /build

COPY Cargo.toml ./
COPY src ./src
COPY include ./include

RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libc6-dev \
        linux-libc-dev \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ccc

COPY --from=builder \
    /build/target/release/ccc \
    /build/target/release/ccc-x86 \
    /build/target/release/ccc-arm \
    /build/target/release/ccc-riscv \
    /build/target/release/ccc-i686 \
    ./bin/
COPY --from=builder /build/include ./include

COPY docker-entrypoint.sh /opt/ccc/docker-entrypoint.sh
RUN chmod +x /opt/ccc/docker-entrypoint.sh

ENV PATH="/opt/ccc/bin:${PATH}"

WORKDIR /work

ENTRYPOINT ["/opt/ccc/docker-entrypoint.sh"]
CMD ["--version"]
