#!/bin/bash
# Voice2Text AI - Flatpak Build and Release Script
# Run this script to build and prepare the flatpak for distribution

set -e  # Exit on any error

echo "🎯 Voice2Text AI - Flatpak Builder"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "com.voice2text.app.yml" ]; then
    print_error "com.voice2text.app.yml not found. Please run this script from the Voice2Text-AI directory."
    exit 1
fi

# Check if flatpak-builder is installed
if ! command -v flatpak-builder &> /dev/null; then
    print_warning "flatpak-builder not found. Installing..."
    sudo apt update && sudo apt install -y flatpak-builder
    print_status "flatpak-builder installed"
else
    print_status "flatpak-builder is available"
fi

# Clean any previous builds
print_status "Cleaning previous builds..."
rm -rf build voice2text.flatpak

# Build the flatpak
print_status "Building Flatpak (this may take several minutes)..."
flatpak-builder --force-clean build com.voice2text.app.yml

# Create distributable bundle
print_status "Creating distributable bundle..."
flatpak build-bundle build voice2text.flatpak com.voice2text.app

# Verify the bundle was created
if [ -f "voice2text.flatpak" ]; then
    BUNDLE_SIZE=$(du -h voice2text.flatpak | cut -f1)
    print_status "Flatpak bundle created: voice2text.flatpak (${BUNDLE_SIZE})"
else
    print_error "Failed to create flatpak bundle"
    exit 1
fi

# Test the bundle locally (optional)
read -p "Do you want to test the flatpak locally? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing locally for testing..."
    flatpak-builder --user --install build com.voice2text.app.yml

    print_status "Testing the app..."
    timeout 10s flatpak run com.voice2text.app || print_warning "App test completed (timeout expected)"
fi

echo ""
print_status "Flatpak build completed successfully!"
echo ""
echo "📦 Next steps:"
echo "1. Create a new GitHub release at: https://github.com/crhy/Voice2Text-AI/releases"
echo "2. Upload the file: voice2text.flatpak"
echo "3. Tag: v0.3-flatpak"
echo "4. Title: Voice2Text AI v0.3 - Linux Flatpak"
echo "5. Description:"
echo '```'
echo "Linux Flatpak distribution of Voice2Text AI."
echo ""
echo "## Installation"
echo '```bash'
echo "# Download voice2text.flatpak from this release"
echo "flatpak install --user voice2text.flatpak"
echo "flatpak run com.voice2text.app"
echo '```'
echo '```'
echo ""
echo "6. Publish the release"
echo "7. Share the download URL with the developer to update README.md"
echo ""
print_status "Build script completed! 🎉"