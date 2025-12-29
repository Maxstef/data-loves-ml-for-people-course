# Project for the Course "Machine Learning for People"

## 1. Overview
This repository contains work, practice exercises, and homework completed as part of the "Machine Learning for People" course.  
More details about the course can be found here: [Data Loves Academy](https://www.dataloves.academy/courses/machine-learning)

---

## 2. Quick Start

Get up and running in just a few steps:

```bash
# 1. Clone the repository
git clone https://github.com/Maxstef/data-loves-ml-for-people-course.git
cd data-loves-ml-for-people-course

# 2. Create and activate the Conda environment
conda create -n mlpeople python=3.11
conda activate mlpeople

# 3. Install dependencies
conda env update -f environment.yml

# 4. Install the project in editable mode
pip install -e .

# 5. (Optional) Install development tools and notebooks
pip install -e ".[dev,notebooks]"

# 6. Launch JupyterLab for notebooks
jupyter lab
```
After this, you can immediately run notebooks, import mlpeople, or start development and testing.

## 3. Project Structure
data-loves-ml-for-people-course/
├── src/
│   └── mlpeople/ # python files orginized as src project (Python package)
│       └── eda/
│           └── categorization/
│           └── missing/
│           └── outliers/
│           └── ...
│       └── io/
│       └── optimization/
│       └── statistics/
│       └── visualization/
│           └── ...
│       └── ...
├── notebooks/ # all practice and homeworks per each module
│   └── 0_python/
│   └── 1_ds_intro/
│       └── ...
│   └── ...
├── tests/ # here can be tests in future
├── environment.yml
├── pyproject.toml
└── README.md

---

## 4. Setup Instructions

### 4.1 Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
- Python >= 3.11 (managed via Conda)

---

### 4.2 Clone the repository
```bash
git clone https://github.com/Maxstef/data-loves-ml-for-people-course.git
cd data-loves-ml-for-people-course
```

### 4.3. Create a dedicated Conda environment

It is **recommended not to use your base environment**.

```bash
# Create the environment
conda create -n mlpeople python=3.11

# Activate it
conda activate mlpeople

# Install all dependencies
conda env update -f environment.yml
```

### 4.4. Install the package in editable mode

```bash
# Install your project locally
pip install -e .
```

### 4.5. Running notebooks
- Install notebook dependencies (if not included):
```bash
pip install -e ".[notebooks]"
```
- To launch JupyterLab:
```bash
jupyter lab
```

### 4.6. Optional: Install development tools

```bash
pip install -e ".[dev]"
```

#### 4.6.1. Running Tests
```bash
pytest -v tests/
```

#### 4.6.2. Format code with Black
Check formatting without changing files:
```bash
black --check src/ tests/
```

Auto-format code:
```bash
black src/ tests/
```

#### 4.6.3. Lint and check code with Ruff
Check for errors, unused imports, and style violations:
```bash
ruff check src/ tests/
```

Automatically fix fixable issues:
```bash
ruff fix src/ tests/
```

#### 4.6.4. Type checking with Mypy
```bash
mypy src/
```