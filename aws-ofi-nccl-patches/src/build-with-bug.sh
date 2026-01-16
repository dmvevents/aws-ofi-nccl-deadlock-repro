#!/bin/bash
# build-with-bug.sh - Build aws-ofi-nccl with specific bugs reintroduced
#
# Usage:
#   ./build-with-bug.sh bug1    # Revert PR #968 (MR deadlock)
#   ./build-with-bug.sh bug3    # Revert PR #1087 (schedule leak)
#   ./build-with-bug.sh all     # Revert all bugs
#   ./build-with-bug.sh none    # Build fixed version (control)
#
# This script must be run inside a container with build tools and libfabric

set -e

BUG_TYPE="${1:-none}"
BUILD_DIR="/tmp/aws-ofi-nccl-build"
INSTALL_PREFIX="/opt/aws-ofi-nccl-${BUG_TYPE}"

# Commit hashes for each bug fix
PR968_COMMIT="06e3ebd1f934bc8334f6fb9b670b573c6b77c6ae"  # MR deadlock fix
PR1087_COMMIT="4071e8b61bc2d1e998dd58900a46bfc8b7ce9fa1" # Schedule leak fix

echo "=========================================="
echo "Building aws-ofi-nccl with bug: ${BUG_TYPE}"
echo "Install prefix: ${INSTALL_PREFIX}"
echo "=========================================="

# Clean and setup
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Clone aws-ofi-nccl
echo "Cloning aws-ofi-nccl..."
git clone https://github.com/aws/aws-ofi-nccl.git
cd aws-ofi-nccl

# Checkout latest version with all fixes
git checkout master
LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "master")
echo "Latest tag: ${LATEST_TAG}"

# Apply reverts based on bug type
case "${BUG_TYPE}" in
    bug1)
        echo "Reverting PR #968 (MR deadlock fix)..."
        if git log --oneline | grep -q "${PR968_COMMIT:0:7}"; then
            git revert --no-commit "${PR968_COMMIT}" || {
                echo "Warning: Could not cleanly revert PR #968"
                echo "Attempting manual revert..."
                # If automatic revert fails, we'll apply manual patch
            }
        else
            echo "Commit ${PR968_COMMIT} not found, checking history..."
            git log --oneline --all | head -20
        fi
        ;;
    bug3)
        echo "Reverting PR #1087 (schedule leak fix)..."
        if git log --oneline | grep -q "${PR1087_COMMIT:0:7}"; then
            git revert --no-commit "${PR1087_COMMIT}" || {
                echo "Warning: Could not cleanly revert PR #1087"
            }
        else
            echo "Commit ${PR1087_COMMIT} not found in history"
            # PR #1087 was merged January 13, 2026, so it should be recent
        fi
        ;;
    all)
        echo "Reverting all bug fixes..."
        for commit in "${PR968_COMMIT}" "${PR1087_COMMIT}"; do
            if git log --oneline | grep -q "${commit:0:7}"; then
                git revert --no-commit "${commit}" || echo "Warning: Could not revert ${commit}"
            fi
        done
        ;;
    none)
        echo "Building fixed version (no reverts)"
        ;;
    v1.14.0)
        echo "Checking out v1.14.0 (original vulnerable version)..."
        git checkout v1.14.0
        ;;
    v1.17.2)
        echo "Checking out v1.17.2..."
        git checkout v1.17.2
        ;;
    *)
        echo "Unknown bug type: ${BUG_TYPE}"
        echo "Valid options: bug1, bug3, all, none, v1.14.0, v1.17.2"
        exit 1
        ;;
esac

# Show what we're building
echo ""
echo "Git status after reverts:"
git status --short
echo ""

# Run autogen
echo "Running autogen..."
./autogen.sh

# Detect container environment
LIBFABRIC_PATH="/opt/amazon/efa"
CUDA_PATH="/usr/local/cuda"
HWLOC_PATH=""

if [ -d "/opt/amazon/efa/lib" ]; then
    echo "Found EFA libfabric at ${LIBFABRIC_PATH}"
fi

if [ -d "/usr/local/cuda" ]; then
    echo "Found CUDA at ${CUDA_PATH}"
fi

# Configure
echo "Configuring..."
./configure \
    --with-libfabric="${LIBFABRIC_PATH}" \
    --with-cuda="${CUDA_PATH}" \
    --enable-platform-aws \
    --prefix="${INSTALL_PREFIX}"

# Build
echo "Building..."
make -j$(nproc)

# Install
echo "Installing to ${INSTALL_PREFIX}..."
make install

# Verify build
echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Library: ${INSTALL_PREFIX}/lib/libnccl-net.so"
ls -la "${INSTALL_PREFIX}/lib/"

# Show version info
echo ""
echo "Version strings:"
strings "${INSTALL_PREFIX}/lib/libnccl-net.so" | grep -iE "version|aws-ofi" | head -5

echo ""
echo "To use this build:"
echo "  export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:\${LD_LIBRARY_PATH}"
