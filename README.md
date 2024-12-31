# Model Interpretability Framework

A framework for analyzing and visualizing deep learning model internals, including layer visualization and feature attribution techniques.

## Features
- Layer-wise activation visualization
- Feature attribution using integrated gradients
- Comprehensive model analysis tools
- Support for PyTorch models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from models.sample_model import SampleModel
from src.interpreter import ModelInterpreter

# Create model and data
model = SampleModel()
sample_input = torch.randn(1, 3, 224, 224)  # Example input

# Initialize interpreter
interpreter = ModelInterpreter(model)

# Run analysis
interpreter.analyze_model(sample_input, target_class=0)
```

## Project Structure
- `data/`: Directory for dataset storage
- `models/`: Model definitions
- `src/`: Core implementation
- `notebooks/`: Example notebooks
- `tests/`: Unit tests

## Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Seaborn# model-interpretabilityy
