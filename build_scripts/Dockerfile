# Use Python 3.12 as base image from Aliyun mirror
FROM m.daocloud.io/docker.io/library/python:3.12-slim

WORKDIR /app

# Route apt through the Aliyun Debian mirror so package fetches succeed in regions
# where deb.debian.org is unreachable. Covers both legacy and deb822 source layouts.
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g; s|http://security.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org|https://mirrors.aliyun.com|g; s|http://security.debian.org|https://mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    fi

# System packages required for building wheels (mysqlclient, sqlite, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc git wget unzip \
        libsqlite3-dev default-libmysqlclient-dev pkg-config vim procps && \
    rm -rf /var/lib/apt/lists/*

# Use Aliyun PyPI mirror for faster installs inside China. The mirror is served
# over HTTPS with a valid certificate, so no `trusted-host` opt-out is needed
# (and adding one would disable TLS/hostname verification for that host).
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# Install datus-agent (fastembed + huggingface_hub come in transitively)
RUN pip install --no-cache-dir datus-agent==0.3.0

# Pin the fastembed cache path so the runtime resolves the same location the
# host prefetch wrote to. HF mirror env vars are retained as a fallback in
# case the runtime ever needs to fetch additional assets.
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    FASTEMBED_CACHE_PATH=/root/.cache/huggingface/fastembed

# Inject the fastembed model cache prefetched on the host
# (build_scripts/prefetch_model.sh writes here). docker_build.sh invokes the
# prefetch automatically when the directory is missing.
COPY build_scripts/cache/fastembed/ /root/.cache/huggingface/fastembed/

# Seed the default agent configuration at the runtime-expected location
RUN mkdir -p /root/.datus/conf
COPY conf/agent.yml.example /root/.datus/conf/agent.yml

WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
