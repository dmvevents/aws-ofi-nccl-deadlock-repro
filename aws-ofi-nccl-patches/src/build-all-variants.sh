#!/bin/bash
# build-all-variants.sh - Build and push all aws-ofi-nccl bug test variants
#
# This script builds Docker images with different aws-ofi-nccl versions:
# - v1.14.0: Original vulnerable version (has Bug₁ MR deadlock)
# - v1.17.2: Your current version (fixed Bug₁, still has Bug₃)
# - v1.17.3/master: Latest with all fixes (control)
#
# Usage:
#   ./build-all-variants.sh           # Build all variants
#   ./build-all-variants.sh v1.14.0   # Build specific variant
#   ./build-all-variants.sh push      # Push to ECR after building

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECR_REPO="058264135704.dkr.ecr.us-east-2.amazonaws.com/deadlock-test"
BASE_IMAGE="${ECR_REPO}:latest"

# Variants to build
VARIANTS=("v1.14.0" "v1.17.2" "none")  # none = latest/fixed

build_variant() {
    local variant=$1
    local tag="aws-ofi-nccl-${variant}"

    echo ""
    echo "=============================================="
    echo "Building variant: ${variant}"
    echo "Tag: ${ECR_REPO}:${tag}"
    echo "=============================================="

    cd "${SCRIPT_DIR}"

    # Build the image
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg BUG_TYPE="${variant}" \
        -t "${ECR_REPO}:${tag}" \
        -f Dockerfile.bug-test \
        .

    echo "Built: ${ECR_REPO}:${tag}"
}

push_variant() {
    local variant=$1
    local tag="aws-ofi-nccl-${variant}"

    echo "Pushing ${ECR_REPO}:${tag}..."
    docker push "${ECR_REPO}:${tag}"
}

# Parse arguments
DO_PUSH=false
SPECIFIC_VARIANT=""

for arg in "$@"; do
    case $arg in
        push)
            DO_PUSH=true
            ;;
        v1.14.0|v1.17.2|none|bug1|bug3|all)
            SPECIFIC_VARIANT=$arg
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [v1.14.0|v1.17.2|none|bug1|bug3|all] [push]"
            exit 1
            ;;
    esac
done

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin "${ECR_REPO%/*}"

# Build variants
if [ -n "${SPECIFIC_VARIANT}" ]; then
    build_variant "${SPECIFIC_VARIANT}"
    if [ "$DO_PUSH" = true ]; then
        push_variant "${SPECIFIC_VARIANT}"
    fi
else
    for variant in "${VARIANTS[@]}"; do
        build_variant "${variant}"
    done

    if [ "$DO_PUSH" = true ]; then
        for variant in "${VARIANTS[@]}"; do
            push_variant "${variant}"
        done
    fi
fi

echo ""
echo "=============================================="
echo "Build Summary"
echo "=============================================="
docker images | grep "${ECR_REPO}" | head -10

echo ""
echo "To run tests:"
echo "  kubectl apply -f pytorchjob-bug-test.yaml"
