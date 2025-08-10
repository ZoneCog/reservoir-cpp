import sys, json, subprocess
from pathlib import Path
from datetime import datetime

def analyze_python_module(py_file):
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        functions, classes = [], []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('def ') and not line.startswith('def _'):
                func_name = line.split('(')[0].replace('def ', '')
                functions.append(func_name)
            elif line.startswith('class '):
                class_name = line.split('(')[0].split(':')[0].replace('class ', '')
                classes.append(class_name)
        return {'functions': functions, 'classes': classes, 'path': str(py_file)}
    except Exception as e:
        return {'functions': [], 'classes': [], 'path': str(py_file), 'error': str(e)}

def check_cpp_equivalent(item_name, cpp_headers_dir, cpp_src_dir):
    for root in [cpp_headers_dir, cpp_src_dir]:
        if Path(root).exists():
            for ext in ['*.hpp', '*.cpp']:
                for file in Path(root).rglob(ext):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            if item_name.lower() in f.read().lower():
                                return True, f"Found in {file.relative_to(Path(root).parent)}"
                    except:
                        continue
    return False, "NOT FOUND"

def main():
    base_dir = Path('.')
    reservoirpy_dir = base_dir / 'TO_REMOVE' / 'reservoirpy'
    cpp_headers_dir = base_dir / 'include'
    cpp_src_dir = base_dir / 'src'
    
    print("=== Detailed Python to C++ Functionality Verification ===")
    print(f"{datetime.now()}")
    
    core_modules = ['activationsfunc.py', 'mat_gen.py', 'node.py', 'model.py', 'ops.py', 'observables.py', 'type.py']
    verification_results = {}
    missing_items = {'functions': [], 'classes': []}
    
    print("\n## Core Module Analysis")
    print("=" * 50)
    
    for module in core_modules:
        module_path = reservoirpy_dir / module
        if module_path.exists():
            print(f"\n### Analyzing {module}")
            analysis = analyze_python_module(module_path)
            
            if 'error' in analysis:
                print(f"❌ Error analyzing {module}: {analysis['error']}")
                continue
            
            print(f"Found {len(analysis['functions'])} functions and {len(analysis['classes'])} classes")
            
            function_results = []
            for func in analysis['functions']:
                has_cpp, location = check_cpp_equivalent(func, cpp_headers_dir, cpp_src_dir)
                function_results.append((func, has_cpp, location))
                status = '✅' if has_cpp else '❌'
                print(f"  Function '{func}': {status} {location}")
                if not has_cpp:
                    missing_items['functions'].append({'name': func, 'module': module})
            
            class_results = []
            for cls in analysis['classes']:
                has_cpp, location = check_cpp_equivalent(cls, cpp_headers_dir, cpp_src_dir)
                class_results.append((cls, has_cpp, location))
                status = '✅' if has_cpp else '❌'
                print(f"  Class '{cls}': {status} {location}")
                if not has_cpp:
                    missing_items['classes'].append({'name': cls, 'module': module})
            
            verification_results[module] = {'functions': function_results, 'classes': class_results}
    
    # Summary
    print("\n## Summary")
    print("=" * 50)
    
    total_functions = sum(len(result['functions']) for result in verification_results.values())
    implemented_functions = sum(1 for result in verification_results.values() 
                              for func, has_cpp, _ in result['functions'] if has_cpp)
    
    total_classes = sum(len(result['classes']) for result in verification_results.values())
    implemented_classes = sum(1 for result in verification_results.values()
                            for cls, has_cpp, _ in result['classes'] if has_cpp)
    
    print(f"Functions: {implemented_functions}/{total_functions} implemented")
    print(f"Classes: {implemented_classes}/{total_classes} implemented")
    print(f"Overall: {implemented_functions + implemented_classes}/{total_functions + total_classes}")
    
    # Output for GitHub Actions
    total_missing = len(missing_items['functions']) + len(missing_items['classes'])
    coverage_percentage = ((implemented_functions + implemented_classes) / (total_functions + total_classes) * 100) if (total_functions + total_classes) > 0 else 100
    
    print(f"\n📊 Coverage: {coverage_percentage:.1f}%")
    print(f"📋 Missing items: {total_missing}")
    
    # Write outputs for GitHub Actions
    with open('missing_functions.json', 'w') as f:
        json.dump(missing_items['functions'], f)
    with open('missing_classes.json', 'w') as f:
        json.dump(missing_items['classes'], f)
    
    # Exit with appropriate code
    success = coverage_percentage >= 90.0
    if success:
        print("\n🎉 High confidence: Most functionality is implemented in C++!")
        return True
    else:
        print("\n⚠️  Some significant functionality may be missing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
