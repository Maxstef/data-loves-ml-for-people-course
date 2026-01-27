"""
Developer utility:
Generates a tree view of the src/ folder including:
- directories
- Python files
- classes and their methods
- top-level functions

Usage:
    python tools/generate_structure.py
"""

import os
import ast
from typing import List, Dict, Tuple

SRC_DIR = "src"

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
}


def extract_symbols(py_file: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract top-level functions and class methods from a Python file.
    """
    with open(py_file, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return [], {}

    functions: List[str] = []
    classes: Dict[str, List[str]] = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)

        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef)
            ]
            classes[node.name] = methods

    return functions, classes


def generate_structure(root_dir: str) -> None:
    """
    Walk through root_dir and print project structure.
    """
    for root, dirs, files in os.walk(root_dir):
        # Filter excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        level = root.replace(root_dir, "").count(os.sep)
        indent = "    " * level
        print(f"{indent}{os.path.basename(root)}/")

        subindent = "    " * (level + 1)

        for file in sorted(files):
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            print(f"{subindent}{file}")

            functions, classes = extract_symbols(file_path)

            for class_name, methods in classes.items():
                print(f"{subindent}    class {class_name}")
                for method in methods:
                    print(f"{subindent}        def {method}()")

            for fn in functions:
                print(f"{subindent}    def {fn}()")


if __name__ == "__main__":
    if not os.path.isdir(SRC_DIR):
        raise RuntimeError(f"Directory '{SRC_DIR}' not found")

    generate_structure(SRC_DIR)
