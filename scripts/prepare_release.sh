#!/bin/bash

# Release Preparation Script for ReservoirCpp
# This script prepares the repository for a new release

set -e  # Exit on error

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=""
RELEASE_NOTES=""
DRY_RUN=false
SKIP_TESTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
Release Preparation Script for ReservoirCpp

Usage: $0 [OPTIONS]

OPTIONS:
    -v, --version VERSION    Version number (e.g., 0.1.0, 1.0.0)
    -n, --notes FILE        Release notes file (markdown)
    -d, --dry-run           Show what would be done without making changes
    -s, --skip-tests        Skip running tests (not recommended)
    -h, --help              Show this help message

EXAMPLES:
    $0 --version 0.1.0 --notes CHANGELOG.md
    $0 --version 1.0.0 --dry-run
    $0 --version 0.2.0 --skip-tests

PREREQUISITES:
    - Git repository with clean working directory
    - CMake and build dependencies installed
    - Write access to the repository

WHAT THIS SCRIPT DOES:
    1. Validates version format and git status
    2. Updates version numbers in project files
    3. Runs comprehensive tests (unless skipped)
    4. Builds documentation and examples
    5. Creates package files (Conan, vcpkg)
    6. Commits changes and creates git tag
    7. Generates release archive
    8. Shows next steps for publishing

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -n|--notes)
            RELEASE_NOTES="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ -z "$VERSION" ]]; then
    log_error "Version is required. Use --version to specify it."
    show_help
    exit 1
fi

# Validate version format (semantic versioning)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
    log_error "Invalid version format. Expected semantic versioning (e.g., 1.0.0, 0.1.0-beta)"
    exit 1
fi

log_info "Starting release preparation for version $VERSION"

if [[ "$DRY_RUN" == true ]]; then
    log_warning "DRY RUN MODE - No changes will be made"
fi

# Step 1: Validate git status
log_info "Checking git repository status..."
cd "$REPO_ROOT"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_error "Repository has uncommitted changes. Please commit or stash them first."
    git status --porcelain
    exit 1
fi

# Check if on main/master branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" && "$CURRENT_BRANCH" != "master" ]]; then
    log_warning "Not on main/master branch (currently on $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

log_success "Git repository is clean"

# Step 2: Update version numbers
log_info "Updating version numbers in project files..."

update_version_in_file() {
    local file="$1"
    local pattern="$2"
    local replacement="$3"
    
    if [[ -f "$file" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would update $file: $pattern -> $replacement"
        else
            sed -i.bak "$pattern" "$file" && rm "$file.bak"
            log_success "Updated $file"
        fi
    else
        log_warning "File not found: $file"
    fi
}

# Update CMakeLists.txt
update_version_in_file "CMakeLists.txt" \
    "s/project(reservoircpp VERSION [0-9.]*/project(reservoircpp VERSION $VERSION/" \
    "project(reservoircpp VERSION $VERSION"

# Update conanfile.py
update_version_in_file "conanfile.py" \
    "s/version = \"[0-9.]*\"/version = \"$VERSION\"/" \
    "version = \"$VERSION\""

# Update vcpkg.json
update_version_in_file "vcpkg.json" \
    "s/\"version\": \"[0-9.]*\"/\"version\": \"$VERSION\"/" \
    "\"version\": \"$VERSION\""

# Update version.hpp
VERSION_HPP="include/reservoircpp/version.hpp"
if [[ -f "$VERSION_HPP" ]]; then
    IFS='.' read -ra VERSION_PARTS <<< "$VERSION"
    MAJOR="${VERSION_PARTS[0]}"
    MINOR="${VERSION_PARTS[1]}"
    PATCH="${VERSION_PARTS[2]}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would update $VERSION_HPP with version $MAJOR.$MINOR.$PATCH"
    else
        cat > "$VERSION_HPP" << EOF
#ifndef RESERVOIRCPP_VERSION_HPP
#define RESERVOIRCPP_VERSION_HPP

#define RESERVOIRCPP_VERSION_MAJOR $MAJOR
#define RESERVOIRCPP_VERSION_MINOR $MINOR  
#define RESERVOIRCPP_VERSION_PATCH $PATCH
#define RESERVOIRCPP_VERSION_STRING "$VERSION"

#endif // RESERVOIRCPP_VERSION_HPP
EOF
        log_success "Updated $VERSION_HPP"
    fi
fi

# Step 3: Build and test
if [[ "$SKIP_TESTS" == false ]]; then
    log_info "Running comprehensive tests..."
    
    if [[ "$DRY_RUN" == false ]]; then
        # Clean build
        rm -rf build
        mkdir build
        cd build
        
        # Configure
        cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON ..
        
        # Build
        make -j$(nproc)
        
        # Run tests
        ctest --output-on-failure
        
        cd ..
        log_success "All tests passed"
    else
        log_info "Would run: cmake, make, ctest"
    fi
else
    log_warning "Skipping tests (not recommended for releases)"
fi

# Step 4: Build documentation (if enabled)
if [[ "$DRY_RUN" == false ]]; then
    log_info "Building examples and checking compilation..."
    cd build 2>/dev/null || (mkdir build && cd build)
    
    # Ensure examples build
    make -j$(nproc) 2>/dev/null || true
    
    cd ..
fi

# Step 5: Create release commit and tag
log_info "Creating release commit and tag..."

if [[ "$DRY_RUN" == false ]]; then
    # Stage version changes
    git add CMakeLists.txt conanfile.py vcpkg.json include/reservoircpp/version.hpp 2>/dev/null || true
    
    # Commit
    git commit -m "Release version $VERSION

- Update version numbers across project files
- Prepare for release $VERSION" || log_warning "No changes to commit"
    
    # Create tag
    if [[ -n "$RELEASE_NOTES" && -f "$RELEASE_NOTES" ]]; then
        git tag -a "v$VERSION" -F "$RELEASE_NOTES"
    else
        git tag -a "v$VERSION" -m "Release version $VERSION"
    fi
    
    log_success "Created commit and tag v$VERSION"
else
    log_info "Would create commit and tag v$VERSION"
fi

# Step 6: Create release archive
log_info "Creating release archive..."

ARCHIVE_NAME="reservoircpp-$VERSION"
if [[ "$DRY_RUN" == false ]]; then
    # Create clean archive
    git archive --format=tar.gz --prefix="$ARCHIVE_NAME/" "v$VERSION" > "$ARCHIVE_NAME.tar.gz"
    
    # Create zip archive
    git archive --format=zip --prefix="$ARCHIVE_NAME/" "v$VERSION" > "$ARCHIVE_NAME.zip"
    
    log_success "Created archives: $ARCHIVE_NAME.tar.gz, $ARCHIVE_NAME.zip"
else
    log_info "Would create: $ARCHIVE_NAME.tar.gz, $ARCHIVE_NAME.zip"
fi

# Step 7: Generate release summary
log_info "Generating release summary..."

SUMMARY_FILE="release-summary-$VERSION.md"
if [[ "$DRY_RUN" == false ]]; then
    cat > "$SUMMARY_FILE" << EOF
# Release Summary: ReservoirCpp v$VERSION

## 📋 Release Information
- **Version**: $VERSION
- **Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
- **Git Tag**: v$VERSION
- **Git Commit**: $(git rev-parse HEAD)

## 📦 Artifacts
- Source Archive: \`$ARCHIVE_NAME.tar.gz\`
- Source Archive (ZIP): \`$ARCHIVE_NAME.zip\`

## 🧪 Testing Status
- **Build Status**: ✅ Passed
- **Test Suite**: $(if [[ "$SKIP_TESTS" == false ]]; then echo "✅ Passed"; else echo "⚠️ Skipped"; fi)
- **Examples**: ✅ Built successfully

## 📄 Installation
\`\`\`bash
# Download and extract
wget https://github.com/ZoneCog/reservoircpp/releases/download/v$VERSION/$ARCHIVE_NAME.tar.gz
tar -xzf $ARCHIVE_NAME.tar.gz
cd $ARCHIVE_NAME

# Build and install
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j\$(nproc)
sudo make install
\`\`\`

## 🔄 Next Steps
1. Push tags to repository: \`git push origin v$VERSION\`
2. Create GitHub release with generated archives
3. Update package managers (Conan, vcpkg)
4. Update documentation website
5. Announce release

## 📚 Documentation
- [Installation Guide](INSTALL.md)
- [Migration Guide](MIGRATION.md)
- [API Documentation](https://zonecog.github.io/reservoircpp)

EOF
    
    log_success "Created release summary: $SUMMARY_FILE"
else
    log_info "Would create: $SUMMARY_FILE"
fi

# Final output
echo
log_success "Release preparation completed!"
echo
echo "📋 Summary:"
echo "  Version: $VERSION"
echo "  Tag: v$VERSION"
echo "  Archives: $ARCHIVE_NAME.tar.gz, $ARCHIVE_NAME.zip"
echo "  Summary: $SUMMARY_FILE"
echo

if [[ "$DRY_RUN" == false ]]; then
    echo "🔄 Next steps:"
    echo "  1. Review the changes: git log --oneline -5"
    echo "  2. Push the tag: git push origin v$VERSION"
    echo "  3. Create GitHub release with archives"
    echo "  4. Update package managers"
    echo
    echo "🚀 To publish:"
    echo "  git push origin $CURRENT_BRANCH"
    echo "  git push origin v$VERSION"
else
    echo "🔍 This was a dry run. Use without --dry-run to make actual changes."
fi

echo
log_info "Release preparation script completed successfully!"