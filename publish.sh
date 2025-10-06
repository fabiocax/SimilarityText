#!/bin/bash

# ==============================================================================
# SimilarityText Publishing Script
# ==============================================================================
#
# This script automates the process of publishing SimilarityText to PyPI.
# It handles building, checking, and uploading the package.
#
# Usage:
#   ./publish.sh [version] [options]
#
# Examples:
#   ./publish.sh 0.3.0              # Publish version 0.3.0
#   ./publish.sh 0.3.0 --test-only  # Test on TestPyPI only
#   ./publish.sh 0.3.0 --skip-test  # Skip TestPyPI, go directly to PyPI
#
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions for colored output
print_header() {
    echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC}  $1"
    echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${CYAN}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Parse command line arguments
VERSION=""
TEST_ONLY=false
SKIP_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./publish.sh [version] [options]"
            echo ""
            echo "Options:"
            echo "  --test-only   Upload to TestPyPI only"
            echo "  --skip-test   Skip TestPyPI, upload directly to PyPI"
            echo "  --help, -h    Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./publish.sh 0.3.0"
            echo "  ./publish.sh 0.3.0 --test-only"
            echo "  ./publish.sh 0.3.0 --skip-test"
            exit 0
            ;;
        *)
            if [[ -z "$VERSION" ]]; then
                VERSION=$1
            else
                print_error "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# ==============================================================================
# Validation
# ==============================================================================

print_header "🚀 SimilarityText Publishing Script"

# Check if version is provided
if [[ -z "$VERSION" ]]; then
    print_error "Version number required"
    echo "Usage: ./publish.sh [version] [options]"
    echo "Example: ./publish.sh 0.3.0"
    exit 1
fi

print_info "Publishing version: ${YELLOW}$VERSION${NC}"

# Check if required tools are installed
print_step "Checking required tools..."

if ! command -v python &> /dev/null; then
    print_error "Python is not installed"
    exit 1
fi
print_success "Python found: $(python --version)"

if ! python -m pip show twine &> /dev/null; then
    print_warning "twine not found, installing..."
    pip install --upgrade twine
fi
print_success "twine installed"

if ! python -m pip show wheel &> /dev/null; then
    print_warning "wheel not found, installing..."
    pip install --upgrade wheel
fi
print_success "wheel installed"

# ==============================================================================
# Version Verification
# ==============================================================================

print_step "Verifying version in setup.py..."

# Extract version from setup.py
SETUP_VERSION=$(grep -oP 'version="\K[^"]+' setup.py)

if [[ "$SETUP_VERSION" != "$VERSION" ]]; then
    print_warning "Version mismatch!"
    print_info "setup.py has version: $SETUP_VERSION"
    print_info "You specified: $VERSION"
    read -p "Update setup.py to version $VERSION? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sed -i "s/version=\"$SETUP_VERSION\"/version=\"$VERSION\"/" setup.py
        print_success "Updated setup.py to version $VERSION"
    else
        print_error "Please update setup.py manually to version $VERSION"
        exit 1
    fi
else
    print_success "Version matches: $VERSION"
fi

# ==============================================================================
# Clean Previous Builds
# ==============================================================================

print_step "Cleaning previous builds..."

rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf similarity.egg-info/
rm -rf SimilarityText.egg-info/

print_success "Build directories cleaned"

# ==============================================================================
# Build Distribution Packages
# ==============================================================================

print_step "Building distribution packages..."

python setup.py sdist bdist_wheel

if [[ ! -d "dist" ]]; then
    print_error "Build failed - dist directory not created"
    exit 1
fi

print_success "Distribution packages built:"
ls -lh dist/

# ==============================================================================
# Check Package Quality
# ==============================================================================

print_step "Checking package quality..."

python -m twine check dist/*

if [[ $? -ne 0 ]]; then
    print_error "Package check failed"
    exit 1
fi

print_success "Package quality check passed"

# ==============================================================================
# Upload to TestPyPI
# ==============================================================================

if [[ "$SKIP_TEST" == false ]]; then
    print_step "Uploading to TestPyPI..."

    read -p "Upload to TestPyPI? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -m twine upload --repository testpypi dist/*

        if [[ $? -eq 0 ]]; then
            print_success "Successfully uploaded to TestPyPI!"
            echo ""
            print_info "Test installation with:"
            echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ SimilarityText"
            echo ""
            print_info "View on TestPyPI:"
            echo "  https://test.pypi.org/project/SimilarityText/"
            echo ""

            if [[ "$TEST_ONLY" == true ]]; then
                print_success "Test-only mode: Stopping here"
                exit 0
            fi

            read -p "Continue to PyPI? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_warning "Stopped before PyPI upload"
                exit 0
            fi
        else
            print_error "Failed to upload to TestPyPI"
            exit 1
        fi
    else
        print_warning "Skipped TestPyPI upload"
    fi
fi

# ==============================================================================
# Upload to PyPI
# ==============================================================================

if [[ "$TEST_ONLY" == false ]]; then
    print_step "Uploading to PyPI (Production)..."

    print_warning "⚠️  This will publish to PRODUCTION PyPI!"
    print_warning "⚠️  This action CANNOT be undone!"
    echo ""
    read -p "Are you absolutely sure? (yes/no) " -r
    echo

    if [[ "$REPLY" == "yes" ]]; then
        python -m twine upload dist/*

        if [[ $? -eq 0 ]]; then
            print_success "🎉 Successfully published to PyPI!"
            echo ""
            print_info "Package URL:"
            echo "  https://pypi.org/project/SimilarityText/"
            echo ""
            print_info "Install with:"
            echo "  pip install SimilarityText"
            echo ""

            # ==============================================================================
            # Git Tag
            # ==============================================================================

            read -p "Create git tag v$VERSION? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                git tag -a "v$VERSION" -m "Release version $VERSION"
                print_success "Git tag v$VERSION created"

                read -p "Push tag to remote? (y/n) " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    git push origin "v$VERSION"
                    print_success "Tag pushed to remote"
                fi
            fi

            # ==============================================================================
            # Final Steps
            # ==============================================================================

            echo ""
            print_header "📋 Next Steps"
            print_info "1. Create GitHub release at:"
            echo "   https://github.com/fabiocax/SimilarityText/releases/new"
            echo ""
            print_info "2. Verify installation:"
            echo "   pip install --upgrade SimilarityText"
            echo ""
            print_info "3. Check PyPI page:"
            echo "   https://pypi.org/project/SimilarityText/"
            echo ""
            print_info "4. Monitor downloads:"
            echo "   https://pypistats.org/packages/similaritytext"
            echo ""
            print_success "✨ Publishing complete!"
        else
            print_error "Failed to upload to PyPI"
            exit 1
        fi
    else
        print_warning "PyPI upload cancelled"
        exit 0
    fi
fi
