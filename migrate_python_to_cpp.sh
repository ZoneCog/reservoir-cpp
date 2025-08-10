#!/bin/bash
# migrate_python_to_cpp.sh
# Script to migrate Python files to TO_REMOVE folder after verification

set -e  # Exit on any error

echo "=== Python to C++ Migration Script ==="
echo "Date: $(date)"
echo ""

# Ensure TO_REMOVE directory exists
mkdir -p TO_REMOVE

# Create backup first
echo "📦 Creating backup of current state..."
if [ ! -d "TO_REMOVE/backup_$(date +%Y%m%d_%H%M%S)" ]; then
    mkdir -p "TO_REMOVE/backup_$(date +%Y%m%d_%H%M%S)"
    cp -r reservoirpy "TO_REMOVE/backup_$(date +%Y%m%d_%H%M%S)/"
    echo "✅ Backup created in TO_REMOVE/backup_$(date +%Y%m%d_%H%M%S)"
fi

# Function to move Python files to TO_REMOVE
move_python_files() {
    local source_dir="$1"
    local target_dir="TO_REMOVE/$1"
    
    if [ -d "$source_dir" ]; then
        echo "📁 Moving $source_dir to $target_dir..."
        
        # Create target directory structure
        mkdir -p "$target_dir"
        
        # Move Python files
        if find "$source_dir" -name "*.py" -type f | grep -q .; then
            find "$source_dir" -name "*.py" -type f -exec mv {} "$target_dir/" \;
            echo "   ✅ Moved Python files from $source_dir"
        else
            echo "   ℹ️  No Python files found in $source_dir"
        fi
        
        # Move Jupyter notebooks  
        if find "$source_dir" -name "*.ipynb" -type f | grep -q .; then
            find "$source_dir" -name "*.ipynb" -type f -exec mv {} "$target_dir/" \;
            echo "   ✅ Moved Jupyter notebooks from $source_dir"
        fi
        
        # Clean up empty directories
        find "$source_dir" -type d -empty -delete 2>/dev/null || true
        
        # If directory is now empty, remove it
        if [ -d "$source_dir" ] && [ ! "$(ls -A "$source_dir")" ]; then
            rmdir "$source_dir"
            echo "   🗑️  Removed empty directory $source_dir"
        fi
    else
        echo "   ⚠️  Directory $source_dir does not exist"
    fi
}

# Function to move specific files
move_specific_files() {
    local file_pattern="$1" 
    local description="$2"
    
    echo "📄 Moving $description..."
    
    if find . -name "$file_pattern" -type f | grep -q .; then
        find . -name "$file_pattern" -type f | while read -r file; do
            target_path="TO_REMOVE/$file"
            mkdir -p "$(dirname "$target_path")"
            mv "$file" "$target_path"
            echo "   ✅ Moved $file to $target_path"
        done
    else
        echo "   ℹ️  No files matching $file_pattern found"
    fi
}

echo "🚀 Starting Python file migration..."
echo ""

# Verify C++ implementation is working first
echo "🔍 Verifying C++ implementation..."
if [ -d "build" ] && [ -f "build/Makefile" ]; then
    cd build
    if make -j$(nproc) > /dev/null 2>&1; then
        echo "✅ C++ build successful"
        
        if ctest --quiet > /dev/null 2>&1; then
            echo "✅ All C++ tests passing ($([[ -f Testing/Temporary/LastTest.log ]] && grep "tests passed" Testing/Temporary/LastTest.log | tail -1 | awk '{print $1}') tests)"
        else
            echo "❌ Some C++ tests failing"
            echo "⚠️  Aborting migration - fix C++ tests first"
            exit 1
        fi
    else
        echo "❌ C++ build failed" 
        echo "⚠️  Aborting migration - fix C++ build first"
        exit 1
    fi
    cd ..
else
    echo "❌ C++ build directory not found"
    echo "⚠️  Aborting migration - build C++ project first"
    exit 1
fi

echo ""
echo "🏃 Proceeding with Python file migration..."
echo ""

# Move core Python modules
echo "== Core Modules =="
move_specific_files "reservoirpy/*.py" "core Python modules"

# Move Python package directories 
echo ""
echo "== Package Directories =="
move_python_files "reservoirpy/nodes"
move_python_files "reservoirpy/datasets" 
move_python_files "reservoirpy/experimental"
move_python_files "reservoirpy/hyper"
move_python_files "reservoirpy/utils"
move_python_files "reservoirpy/compat"
move_python_files "reservoirpy/tests"

# Move any remaining Python files
echo ""
echo "== Remaining Python Files =="
move_specific_files "*.py" "remaining Python files"
move_specific_files "**/*.py" "nested Python files"

# Move Python-specific configuration files
echo ""
echo "== Python Configuration =="
move_specific_files "setup.py" "Python setup files"
move_specific_files "setup.cfg" "Python setup config"
move_specific_files "pyproject.toml" "Python project config"  
move_specific_files "Pipfile" "Python Pipfile"
move_specific_files "requirements*.txt" "Python requirements"

# Move Jupyter notebooks from tutorials and examples
echo ""
echo "== Jupyter Notebooks ==" 
move_python_files "tutorials"
find examples -name "*.ipynb" -type f | while read -r notebook; do
    target_path="TO_REMOVE/$notebook"
    mkdir -p "$(dirname "$target_path")"
    mv "$notebook" "$target_path"
    echo "   ✅ Moved $notebook to $target_path"
done

# Clean up empty Python directories
echo ""
echo "🧹 Cleaning up empty directories..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -type d -empty -delete 2>/dev/null || true

echo ""
echo "📊 Migration Summary"
echo "===================="

# Count moved files
py_files=$(find TO_REMOVE -name "*.py" -type f | wc -l)
ipynb_files=$(find TO_REMOVE -name "*.ipynb" -type f | wc -l) 
config_files=$(find TO_REMOVE -name "setup.py" -o -name "setup.cfg" -o -name "pyproject.toml" -o -name "Pipfile" -o -name "requirements*.txt" | wc -l)

echo "📄 Python files moved: $py_files"
echo "📓 Jupyter notebooks moved: $ipynb_files"
echo "⚙️  Configuration files moved: $config_files"
echo "📁 Total files in TO_REMOVE: $(find TO_REMOVE -type f | wc -l)"

echo ""
echo "✅ Migration completed successfully!"
echo "🎉 Repository converted to pure C++ implementation"
echo ""
echo "📝 Next steps:"
echo "   1. Verify C++ build still works: cd build && make && ctest"
echo "   2. Review moved files in TO_REMOVE/"
echo "   3. Update documentation to reflect pure C++ implementation"
echo "   4. Commit changes to version control"
