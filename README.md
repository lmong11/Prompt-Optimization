# LLM Prompt Selection System using Coalition Game Theory

This repository contains a system for selecting optimal prompts for Large Language Models (LLMs) using a coalition game theory approach. The system can evaluate prompt performance, train utility models, and select the best prompt or combination of prompts for specific tasks.

## Overview

Traditional LLM prompt engineering relies on manually crafted prompts for specific tasks. This project provides an automated approach that:

1. Evaluates prompt performance across different task categories
2. Trains a utility function model to predict prompt effectiveness
3. Uses coalition game theory to identify optimal combinations of prompts
4. Selects the best prompt or prompt coalition for a given task in real-time

## Features

- **Prompt Repository Management**: Load and manage a diverse collection of system prompts
- **Task Example Generation**: Create evaluation examples from real datasets across multiple categories
- **Performance Evaluation**: Assess prompt performance on various tasks with or without API calls
- **Utility Model Training**: Train models to predict prompt effectiveness for different tasks
- **Coalition Game Theory**: Find optimal prompt combinations using Shapley values
- **Interactive Metrics Dashboard**: Visualize and analyze prompt performance data
- **Command Line Interface**: Easy-to-use commands for all system functions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-prompt-selection.git
cd llm-prompt-selection

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- sentence-transformers
- tqdm
- openai (optional, for real evaluation)
- matplotlib
- seaborn
- dash (for interactive dashboard)
- plotly

## Usage

### 1. Data Preparation

Process your prompts and create evaluation datasets:

```bash
python main.py process_data --csv System_Prompt.csv --output data
```

### 2. Evaluating Prompts

You can either simulate evaluations or use a real LLM API:

```bash
# Simulate evaluations
python main.py simulate --data_dir data

# Or with real LLM (requires API key)
python main.py pipeline --eval --api_key YOUR_OPENAI_API_KEY
```

### 3. Training the Utility Model

Train the utility function model using evaluation results:

```bash
python main.py train --data_dir data --model_dir models
```

### 4. Selecting Prompts for Tasks

Use the model to select optimal prompts for specific tasks:

```bash
# Select a single prompt
python main.py select --task "Write a function to find duplicates in an array" --csv System_Prompt.csv --data_dir data --model_dir models

# Select a coalition of prompts
python main.py select --task "Write a function to find duplicates in an array" --csv System_Prompt.csv --data_dir data --model_dir models --coalition --max_size 3
```

### 5. Analyzing Coalition Performance

Analyze and visualize the performance improvements from using prompt coalitions:

```bash
# Quick analysis without API calls
python analyze_coalitions.py --csv System_Prompt.csv --data_dir data --model_dir models --examples 5

# Comprehensive evaluation with visualization
python coalition_metrics.py --csv System_Prompt.csv --data_dir data --model_dir models --test_size 20

# Interactive dashboard
python coalition_dashboard.py --results_dir results
```

### 6. Running the Complete Pipeline

Run the complete end-to-end pipeline:

```bash
python main.py pipeline --csv System_Prompt.csv --data_dir data --model_dir models
```

## Project Structure

- `data_preparation.py`: Dataset loading and preparation
- `prompt_evaluator.py`: Prompt performance evaluation
- `utility_model.py`: Utility function model and coalition game logic
- `prompt_selection_system.py`: End-to-end system for prompt selection
- `main.py`: Command line interface
- `coalition_metrics.py`: Performance analysis for prompt coalitions
- `analyze_coalitions.py`: Quick coalition analysis tool
- `coalition_dashboard.py`: Interactive metrics dashboard

## Creating Your Own Prompt CSV

Your `System_Prompt.csv` should have the following columns:
- Prompt ID: Unique identifier (e.g., CG-01)
- Prompt Name: Descriptive name
- Full Prompt Text: The complete system prompt
- Intended Function: Description of the prompt's purpose
- Example Inputs: Sample tasks the prompt is designed to handle
- Expected Output Style: Description of expected outputs
- Strengths: The prompt's strengths
- Limitations: Known limitations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project utilizes sentence transformers for semantic embeddings
- Coalition game theory concepts are implemented for prompt combinations
- Evaluation uses benchmark datasets from various sources
