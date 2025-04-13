#!/usr/bin/env python3
"""
LLM Prompt Selection System using Coalition Game Theory

This script provides functionality to:
1. Process the System_Prompt.csv file and create evaluation datasets using real datasets
2. Train a utility function model on evaluation results
3. Select optimal prompts or prompt coalitions for given tasks
4. Run the complete end-to-end pipeline
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json

# Import local modules
from data_preparation import PromptDataProcessor, RealDatasetManager
from prompt_evaluator import PromptEvaluator
from utility_model import UtilityFunctionModel, CoalitionGamePromptSelector
from prompt_selection_system import PromptSelectionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prompt_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_data(prompts_csv: str, output_dir: str = "data") -> None:
    """
    Process the prompts CSV file and create evaluation datasets using real datasets
    
    Args:
        prompts_csv: Path to the CSV file with system prompts
        output_dir: Directory to store processed data
    """
    processor = PromptDataProcessor(prompts_csv, output_dir=output_dir)
    logger.info(f"Loading prompts from {prompts_csv}")
    prompts_df = processor.load_prompts()
    
    logger.info(f"Generating task examples using real datasets")
    examples_df = processor.generate_task_examples()
    
    logger.info(f"Creating evaluation pairs")
    eval_pairs_df = processor.create_evaluation_pairs()
    
    logger.info(f"Preparing train/test split")
    train_df, test_df = processor.prepare_training_data()
    
    logger.info(f"Data processing complete. Files saved to {output_dir}/")
    print(f"Prompts: {len(prompts_df)}")
    print(f"Task examples: {len(examples_df)}")
    print(f"Evaluation pairs: {len(eval_pairs_df)}")
    print(f"Train set: {len(train_df)}")
    print(f"Test set: {len(test_df)}")
    
    # Save a sample of examples by dataset source
    if 'source' in examples_df.columns:
        sample_by_source = {}
        for source in examples_df['source'].unique():
            source_examples = examples_df[examples_df['source'] == source].sample(
                min(3, len(examples_df[examples_df['source'] == source]))
            )
            sample_by_source[source] = source_examples['task_text'].tolist()
        
        with open(os.path.join(output_dir, "sample_examples_by_source.json"), 'w') as f:
            json.dump(sample_by_source, f, indent=2)
        
        # Print some sample examples
        print("\nSample examples from datasets:")
        for source, examples in sample_by_source.items():
            print(f"\n{source}:")
            for i, example in enumerate(examples):
                print(f"  {i+1}. {example[:100]}...")

def simulate_evaluations(data_dir: str = "data") -> None:
    """
    Simulate prompt evaluations for training purpose
    
    Args:
        data_dir: Directory with processed data
    """
    try:
        train_df = pd.read_csv(f"{data_dir}/train_data.csv")
    except FileNotFoundError:
        logger.error(f"Training data not found. Run process_data first.")
        return
    
    # Create a copy to avoid modifying the original
    simulated_results = train_df.copy()
    
    # Baseline random scores
    simulated_results['score'] = np.random.uniform(0.3, 0.7, size=len(simulated_results))
    
    # Assign higher scores when prompt and task categories match (simulating better performance)
    mask = simulated_results['prompt_category'] == simulated_results['task_category']
    simulated_results.loc[mask, 'score'] = np.random.uniform(0.7, 0.95, size=mask.sum())
    
    # Add some noise based on 'relevance' column if it exists
    if 'relevance' in simulated_results.columns:
        simulated_results['score'] = simulated_results['score'] * (0.8 + 0.2 * simulated_results['relevance'])
    
    simulated_results['score'] = simulated_results['score'].clip(0, 1)  # Ensure scores are in [0,1]
    
    # Save simulated results
    output_path = f"{data_dir}/simulated_evaluation_results.csv"
    simulated_results.to_csv(output_path, index=False)
    
    logger.info(f"Simulated evaluation results saved to {output_path}")
    print(f"Total evaluations: {len(simulated_results)}")
    print(f"Average score: {simulated_results['score'].mean():.2f}")
    print(f"Category match average: {simulated_results.loc[mask, 'score'].mean():.2f}")
    print(f"Non-match average: {simulated_results.loc[~mask, 'score'].mean():.2f}")

def train_model(data_dir: str = "data", model_dir: str = "models") -> None:
    """
    Train the utility function model on evaluation results
    
    Args:
        data_dir: Directory with evaluation data
        model_dir: Directory to store trained models
    """
    # Try to load evaluation results (real or simulated)
    try:
        results_path = f"{data_dir}/evaluation_results.csv"
        results_df = pd.read_csv(results_path)
        logger.info(f"Loaded evaluation results from {results_path}")
    except FileNotFoundError:
        try:
            results_path = f"{data_dir}/simulated_evaluation_results.csv"
            results_df = pd.read_csv(results_path)
            logger.info(f"Loaded simulated evaluation results from {results_path}")
        except FileNotFoundError:
            logger.error(f"No evaluation results found. Run evaluate_prompts or simulate_evaluations first.")
            return
    
    # Initialize and train the model
    utility_model = UtilityFunctionModel(model_dir=model_dir)
    
    # Split data if not already split
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(results_df, test_size=0.2, random_state=42)
    
    logger.info(f"Training utility model on {len(train_data)} examples")
    metrics = utility_model.train(train_data, val_data)
    
    logger.info(f"Model training complete. Files saved to {model_dir}/")
    print(f"Training metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def select_prompt(
    task_text: str,
    prompts_csv: str = "System_Prompt.csv",
    data_dir: str = "data",
    model_dir: str = "models",
    coalition: bool = False,
    max_size: int = 3
) -> None:
    """
    Select optimal prompt(s) for a given task
    
    Args:
        task_text: The task text
        prompts_csv: Path to CSV file with system prompts
        data_dir: Directory with processed data
        model_dir: Directory with trained models
        coalition: Whether to use coalition game approach
        max_size: Maximum coalition size
    """
    # Initialize system
    system = PromptSelectionSystem(
        prompts_csv=prompts_csv,
        data_dir=data_dir,
        model_dir=model_dir
    )
    
    # Select prompt(s)
    result = system.select_optimal_prompt(
        task_text=task_text,
        coalition_mode=coalition,
        max_coalition_size=max_size
    )
    
    # Print results
    print(f"\nTask: {task_text}")
    print(f"Inferred category: {result['task_category']}")
    
    if coalition:
        print("\nSelected prompt coalition:")
        for i, prompt in enumerate(result['selected_prompts']):
            print(f"{i+1}. {prompt['name']} ({prompt['id']})")
        
        print("\nCombined prompt:")
        print("-" * 50)
        print(result['combined_prompt'])
        print("-" * 50)
    else:
        print(f"\nSelected prompt: {result['prompt_name']} ({result['prompt_id']})")
        print("\nPrompt text:")
        print("-" * 50)
        print(result['prompt_text'])
        print("-" * 50)

def evaluate_system(
    prompts_csv: str = "System_Prompt.csv",
    data_dir: str = "data",
    model_dir: str = "models",
    num_examples: int = 10
) -> None:
    """
    Evaluate the prompt selection system on test examples
    
    Args:
        prompts_csv: Path to CSV file with system prompts
        data_dir: Directory with processed data
        model_dir: Directory with trained models
        num_examples: Number of test examples to evaluate
    """
    # Initialize system
    system = PromptSelectionSystem(
        prompts_csv=prompts_csv,
        data_dir=data_dir,
        model_dir=model_dir
    )
    
    # Load test data
    try:
        test_df = pd.read_csv(f"{data_dir}/test_data.csv")
        logger.info(f"Loaded test data with {len(test_df)} examples")
    except FileNotFoundError:
        logger.error(f"Test data not found. Run process_data first.")
        return
    
    # Sample test examples
    test_sample = test_df.sample(min(num_examples, len(test_df)), random_state=42)
    
    # Evaluate on test examples
    results = []
    
    for i, row in test_sample.iterrows():
        task_text = row['task_text']
        task_category = row['task_category']
        
        logger.info(f"Evaluating example {i+1}/{len(test_sample)}: {task_text[:50]}...")
        
        # Select individual prompt
        individual_result = system.select_optimal_prompt(
            task_text=task_text,
            task_category=task_category,
            coalition_mode=False
        )
        
        # Select coalition
        coalition_result = system.select_optimal_prompt(
            task_text=task_text,
            task_category=task_category,
            coalition_mode=True,
            max_coalition_size=3
        )
        
        # Store results
        results.append({
            'task_id': row['task_id'],
            'task_text': task_text,
            'task_category': task_category,
            'individual_prompt': individual_result['prompt_id'],
            'coalition_prompts': ','.join([p['id'] for p in coalition_result['selected_prompts']])
        })
        
        # Print results for this example
        print(f"\nExample {i+1}:")
        print(f"Task: {task_text[:100]}{'...' if len(task_text) > 100 else ''}")
        print(f"Category: {task_category}")
        print(f"Best individual prompt: {individual_result['prompt_id']} ({individual_result['prompt_name']})")
        print(f"Optimal coalition: {[p['id'] for p in coalition_result['selected_prompts']]}")
    
    # Save evaluation results
    results_df = pd.DataFrame(results)
    output_path = f"{data_dir}/system_evaluation.csv"
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"System evaluation complete. Results saved to {output_path}")
    print(f"\nEvaluation complete. Results saved to {output_path}")

def run_pipeline(
    prompts_csv: str = "System_Prompt.csv",
    data_dir: str = "data",
    model_dir: str = "models",
    eval_mode: bool = False,
    sample_size: int = 100,
    api_key: Optional[str] = None
) -> None:
    """
    Run the complete end-to-end pipeline
    
    Args:
        prompts_csv: Path to CSV file with system prompts
        data_dir: Directory for processed data
        model_dir: Directory for trained models
        eval_mode: Whether to use real evaluation (requires API key)
        sample_size: Number of samples to evaluate
        api_key: OpenAI API key (required if eval_mode is True)
    """
    if eval_mode and api_key is None:
        logger.error("API key is required for evaluation mode")
        return
    
    # Initialize system
    system = PromptSelectionSystem(
        prompts_csv=prompts_csv,
        data_dir=data_dir,
        model_dir=model_dir,
        openai_api_key=api_key
    )
    
    # Run the pipeline
    logger.info(f"Running end-to-end pipeline with eval_mode={eval_mode}")
    system.run_end_to_end_pipeline(
        sample_size=sample_size,
        eval_mode=eval_mode
    )
    
    logger.info(f"Pipeline completed successfully")
    print(f"Pipeline completed successfully")

def list_datasets() -> None:
    """List available datasets for task examples"""
    # Create a temporary dataset manager
    dataset_manager = RealDatasetManager()
    
    print("\nAvailable datasets by category:")
    for category, datasets in dataset_manager.datasets.items():
        print(f"\n{category}:")
        for dataset in datasets:
            print(f"  - {dataset['name']}: {dataset['url']}")

def main():
    parser = argparse.ArgumentParser(description="LLM Prompt Selection System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process data command
    process_parser = subparsers.add_parser("process_data", help="Process prompt CSV and create datasets")
    process_parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    process_parser.add_argument("--output", type=str, default="data", help="Output directory")
    
    # Simulate evaluations command
    simulate_parser = subparsers.add_parser("simulate", help="Simulate prompt evaluations")
    simulate_parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train utility function model")
    train_parser.add_argument("--data_dir", type=str, default="data", help="Data directory with evaluation results")
    train_parser.add_argument("--model_dir", type=str, default="models", help="Directory to store models")
    
    # Select prompt command
    select_parser = subparsers.add_parser("select", help="Select prompt for a task")
    select_parser.add_argument("--task", type=str, required=True, help="Task text")
    select_parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    select_parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    select_parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    select_parser.add_argument("--coalition", action="store_true", help="Use coalition game approach")
    select_parser.add_argument("--max_size", type=int, default=3, help="Maximum coalition size")
    
    # Evaluate system command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate prompt selection system")
    eval_parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    eval_parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    eval_parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    eval_parser.add_argument("--num", type=int, default=10, help="Number of test examples")
    
    # Run pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run end-to-end pipeline")
    pipeline_parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    pipeline_parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    pipeline_parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    pipeline_parser.add_argument("--eval", action="store_true", help="Use real evaluation")
    pipeline_parser.add_argument("--sample_size", type=int, default=100, help="Evaluation sample size")
    pipeline_parser.add_argument("--api_key", type=str, help="OpenAI API key (required if --eval is used)")
    
    # List datasets command
    list_parser = subparsers.add_parser("list_datasets", help="List available datasets")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    if args.command in ["process_data", "pipeline"]:
        output_dir = args.output if args.command == "process_data" else args.data_dir
        os.makedirs(output_dir, exist_ok=True)
        if args.command == "pipeline":
            os.makedirs(args.model_dir, exist_ok=True)
    
    # Execute command
    if args.command == "process_data":
        process_data(args.csv, args.output)
    elif args.command == "simulate":
        simulate_evaluations(args.data_dir)
    elif args.command == "train":
        train_model(args.data_dir, args.model_dir)
    elif args.command == "select":
        select_prompt(args.task, args.csv, args.data_dir, args.model_dir, args.coalition, args.max_size)
    elif args.command == "evaluate":
        evaluate_system(args.csv, args.data_dir, args.model_dir, args.num)
    elif args.command == "pipeline":
        run_pipeline(args.csv, args.data_dir, args.model_dir, args.eval, args.sample_size, args.api_key)
    elif args.command == "list_datasets":
        list_datasets()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()