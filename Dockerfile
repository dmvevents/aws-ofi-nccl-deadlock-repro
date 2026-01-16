# Dockerfile for aws-ofi-nccl Deadlock Reproduction Test
#
# Builds a container with both vulnerable (1.14.0) and fixed (1.17.2)
# versions of aws-ofi-nccl for comparative testing.
#
# The NGC NeMo container has a known issue where /etc/ld.so.conf.d/aws-ofi-nccl.conf
# forces loading the old library. This Dockerfile handles that by:
#   1. Preserving the old library for testing
#   2. Removing the ldconfig override
#   3. Building the fixed version fresh
#
# Build:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/nemo:25.09.00 \
#       -t aws-ofi-nccl-deadlock-test:latest .

ARG BASE_IMAGE=nvcr.io/nvidia/nemo:25.09.00
FROM ${BASE_IMAGE}

USER root

# ============================================================
# STEP 1: Diagnose base image
# ============================================================
RUN echo "=== BASE IMAGE DIAGNOSIS ===" && \
    echo "OS:" && cat /etc/os-release | grep -E "^(NAME|VERSION)=" && \
    echo "CUDA:" && ls -d /usr/local/cuda* 2>/dev/null | head -1 && \
    echo "EFA:" && cat /opt/amazon/efa/version.txt 2>/dev/null || echo "Unknown" && \
    echo "Existing aws-ofi-nccl:" && ls -la /opt/amazon/aws-ofi-nccl/lib/ 2>/dev/null || echo "Not found" && \
    echo "ldconfig override:" && cat /etc/ld.so.conf.d/aws-ofi-nccl.conf 2>/dev/null || echo "None"

# ============================================================
# STEP 2: Preserve old 1.14.0 for vulnerability testing
# ============================================================
RUN echo "=== PRESERVING VULNERABLE LIBRARY ===" && \
    if [ -d /opt/amazon/aws-ofi-nccl ]; then \
        mv /opt/amazon/aws-ofi-nccl /opt/aws-ofi-nccl-1.14.0-vulnerable && \
        rm -f /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
        ldconfig && \
        echo "Preserved at /opt/aws-ofi-nccl-1.14.0-vulnerable"; \
    else \
        mkdir -p /opt/aws-ofi-nccl-1.14.0-vulnerable; \
    fi

# ============================================================
# STEP 3: Install build dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    libhwloc-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# STEP 4: Build aws-ofi-nccl 1.17.2 (FIXED version)
# ============================================================
ARG OFI_NCCL_VERSION=1.17.2
WORKDIR /tmp/build

RUN CUDA_PATH=$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 || echo "/usr/local/cuda") && \
    EFA_PATH=/opt/amazon/efa && \
    echo "Building aws-ofi-nccl $OFI_NCCL_VERSION..." && \
    echo "  CUDA: $CUDA_PATH" && \
    echo "  EFA: $EFA_PATH" && \
    git clone --depth 1 --branch v${OFI_NCCL_VERSION} https://github.com/aws/aws-ofi-nccl.git && \
    cd aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure \
        --with-libfabric=$EFA_PATH \
        --with-cuda=$CUDA_PATH \
        --prefix=/opt/amazon/aws-ofi-nccl \
        --disable-werror && \
    make -j$(nproc) && \
    make install && \
    echo "Fixed library installed at /opt/amazon/aws-ofi-nccl"

# ============================================================
# STEP 5: Build 1.14.0 if not preserved from base image
# ============================================================
RUN if [ ! -f /opt/aws-ofi-nccl-1.14.0-vulnerable/lib/libnccl-net.so ]; then \
        CUDA_PATH=$(ls -d /usr/local/cuda-* 2>/dev/null | head -1 || echo "/usr/local/cuda") && \
        EFA_PATH=/opt/amazon/efa && \
        echo "Building aws-ofi-nccl 1.14.0 (vulnerable)..." && \
        cd /tmp/build && rm -rf aws-ofi-nccl && \
        git clone --depth 1 --branch v1.14.0 https://github.com/aws/aws-ofi-nccl.git && \
        cd aws-ofi-nccl && \
        ./autogen.sh && \
        ./configure \
            --with-libfabric=$EFA_PATH \
            --with-cuda=$CUDA_PATH \
            --prefix=/opt/aws-ofi-nccl-1.14.0-vulnerable \
            --disable-werror && \
        make -j$(nproc) && \
        make install; \
    fi

RUN rm -rf /tmp/build

# ============================================================
# STEP 6: Build error injection library
# ============================================================
COPY src/inject_mr_errors.c /tmp/
RUN gcc -shared -fPIC -O2 -o /opt/inject_mr_errors.so /tmp/inject_mr_errors.c -ldl -lpthread && \
    rm /tmp/inject_mr_errors.c

# ============================================================
# STEP 7: Copy test scripts
# ============================================================
RUN mkdir -p /opt/tests
COPY src/moe_stress_test.py /opt/tests/
COPY src/entrypoint.sh /opt/tests/
RUN chmod +x /opt/tests/*.sh /opt/tests/*.py

# ============================================================
# STEP 8: Configure library paths
# ============================================================
RUN echo "/opt/amazon/efa/lib" > /etc/ld.so.conf.d/efa.conf && \
    echo "# Library selection via LD_LIBRARY_PATH at runtime" > /etc/ld.so.conf.d/aws-ofi-nccl.conf && \
    ldconfig

# Store paths for runtime
RUN echo "VULNERABLE_LIB=/opt/aws-ofi-nccl-1.14.0-vulnerable" > /opt/library_paths.env && \
    echo "FIXED_LIB=/opt/amazon/aws-ofi-nccl" >> /opt/library_paths.env && \
    echo "EFA_LIB=/opt/amazon/efa" >> /opt/library_paths.env

# ============================================================
# STEP 9: Final verification
# ============================================================
RUN echo "============================================================" && \
    echo "BUILD COMPLETE" && \
    echo "============================================================" && \
    echo "VULNERABLE (1.14.0):" && ls -la /opt/aws-ofi-nccl-1.14.0-vulnerable/lib/libnccl-net.so 2>/dev/null || echo "  Not found" && \
    echo "FIXED (1.17.2):" && ls -la /opt/amazon/aws-ofi-nccl/lib/libnccl-net.so && \
    echo "EFA:" && cat /opt/amazon/efa/version.txt 2>/dev/null && \
    echo "Injector:" && ls -la /opt/inject_mr_errors.so && \
    echo "============================================================" && \
    echo "Usage:" && \
    echo "  LIBRARY_MODE=vulnerable  -> Uses 1.14.0 (has deadlock bug)" && \
    echo "  LIBRARY_MODE=fixed       -> Uses 1.17.2 (deadlock fixed)" && \
    echo "============================================================"

WORKDIR /opt/tests
ENTRYPOINT ["/bin/bash", "/opt/tests/entrypoint.sh"]
