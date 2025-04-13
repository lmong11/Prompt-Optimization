#!/usr/bin/env python3
"""
Quick Coalition Performance Analysis Script

This script provides a quick analysis of coalition performance improvements
without requiring full evaluation. It uses the trained utility model to estimate
improvements across different task categories.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import argparse
import json

# Import from main project
from prompt_selection_system import PromptSelectionSystem
from data_preparation import PromptDataProcessor

def analyze_coalition_performance(
    prompts_csv: str = "System_Prompt.csv",
    data_dir: str = "data",
    model_dir: str = "models",
    output_dir: str = "results",
    examples_per_category: int = 5,
    coalition_sizes: List[int] = [2, 3],
    random_seed: int = 42
):
    """
    Analyze the performance improvement from using prompt coalitions versus individual prompts
    
    Args:
        prompts_csv: Path to prompts CSV file
        data_dir: Data directory with processed data
        model_dir: Directory with trained models
        output_dir: Directory for output results
        examples_per_category: Number of examples to analyze per task category
        coalition_sizes: List of coalition sizes to analyze
        random_seed: Random seed for reproducibility
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize prompt selection system
    system = PromptSelectionSystem(
        prompts_csv=prompts_csv,
        data_dir=data_dir,
        model_dir=model_dir
    )
    
    # Load test data
    try:
        test_df = pd.read_csv(f"{data_dir}/test_data.csv")
        print(f"Loaded test data with {len(test_df)} examples")
    except FileNotFoundError:
        print(f"Test data not found. Run process_data first.")
        return
    
    # Get unique task categories
    task_categories = test_df['task_category'].unique()
    
    all_results = []
    category_samples = {}
    
    # Create stratified sample with equal examples per category
    np.random.seed(random_seed)
    for category in task_categories:
        category_df = test_df[test_df['task_category'] == category]
        if len(category_df) > 0:
            sample_size = min(examples_per_category, len(category_df))
            category_samples[category] = category_df.sample(sample_size)
    
    # Process each test example
    for category, samples in category_samples.items():
        print(f"\nAnalyzing {len(samples)} examples for category {category}...")
        
        for i, (_, row) in enumerate(samples.iterrows()):
            task_text = row['task_text']
            task_id = row['task_id']
            
            print(f"  Example {i+1}: {task_text[:50]}...")
            
            # Get best individual prompt
            individual_result = system.select_optimal_prompt(
                task_text=task_text,
                task_category=category,
                coalition_mode=False
            )
            
            # Use utility model to predict individual score
            individual_data = pd.DataFrame([{
                'prompt_text': individual_result['prompt_text'],
                'prompt_category': individual_result.get('prompt_category', category),
                'task_text': task_text,
                'task_category': category
            }])
            
            individual_score = system.utility_model.predict(individual_data)[0]
            
            # Base result for this task
            base_result = {
                'task_id': task_id,
                'task_category': category,
                'individual_prompt_id': individual_result['prompt_id'],
                'individual_score': individual_score
            }
            
            # Get coalition results for different sizes
            for size in coalition_sizes:
                print(f"    Testing coalition size {size}...")
                
                # Get coalition
                coalition_result = system.select_optimal_prompt(
                    task_text=task_text,
                    task_category=category,
                    coalition_mode=True,
                    max_coalition_size=size
                )
                
                # Skip if coalition has only one prompt (same as individual)
                if len(coalition_result['selected_prompts']) <= 1:
                    continue
                
                # Use utility model to predict coalition score
                coalition_data = pd.DataFrame([{
                    'prompt_text': coalition_result['combined_prompt'],
                    'prompt_category': '+'.join([p['category'] for p in coalition_result['selected_prompts']]),
                    'task_text': task_text,
                    'task_category': category
                }])
                
                coalition_score = system.utility_model.predict(coalition_data)[0]
                
                # Create result for this coalition
                result = base_result.copy()
                result.update({
                    'coalition_size': len(coalition_result['selected_prompts']),
                    'coalition_ids': ','.join([p['id'] for p in coalition_result['selected_prompts']]),
                    'coalition_score': coalition_score,
                    'improvement': coalition_score - individual_score,
                    'percent_improvement': (
                        (coalition_score - individual_score) / max(individual_score, 0.001) * 100
                    )
                })
                
                all_results.append(result)
                
                print(f"    Improvement: {result['improvement']:.4f} ({result['percent_improvement']:.2f}%)")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "coalition_performance.csv"), index=False)
    print(f"\nSaved detailed results to {os.path.join(output_dir, 'coalition_performance.csv')}")
    
    # Calculate summary metrics
    summary = calculate_summary_metrics(results_df)
    
    # Save summary
    with open(os.path.join(output_dir, "coalition_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to {os.path.join(output_dir, 'coalition_summary.json')}")
    
    # Generate visualizations
    create_visualizations(results_df, output_dir)
    print(f"Saved visualizations to {output_dir}")
    
    # Print summary report
    print_summary_report(summary)
    
    return results_df, summary

def calculate_summary_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics from results
    
    Args:
        results_df: DataFrame with analysis results
        
    Returns:
        Dictionary with summary metrics
    """
    # Overall metrics
    overall = {
        'avg_individual_score': results_df['individual_score'].mean(),
        'avg_coalition_score': results_df['coalition_score'].mean(),
        'avg_improvement': results_df['improvement'].mean(),
        'avg_percent_improvement': results_df['percent_improvement'].mean(),
        'median_improvement': results_df['improvement'].median(),
        'positive_improvements': (results_df['improvement'] > 0).sum(),
        'negative_improvements': (results_df['improvement'] < 0).sum(),
        'total_examples': len(results_df)
    }
    
    # Metrics by category
    by_category = {}
    for category in results_df['task_category'].unique():
        cat_df = results_df[results_df['task_category'] == category]
        by_category[category] = {
            'avg_individual_score': cat_df['individual_score'].mean(),
            'avg_coalition_score': cat_df['coalition_score'].mean(),
            'avg_improvement': cat_df['improvement'].mean(),
            'avg_percent_improvement': cat_df['percent_improvement'].mean(),
            'count': len(cat_df)
        }
    
    # Metrics by coalition size
    by_size = {}
    for size in results_df['coalition_size'].unique():
        size_df = results_df[results_df['coalition_size'] == size]
        by_size[str(size)] = {
            'avg_individual_score': size_df['individual_score'].mean(),
            'avg_coalition_score': size_df['coalition_score'].mean(),
            'avg_improvement': size_df['improvement'].mean(),
            'avg_percent_improvement': size_df['percent_improvement'].mean(),
            'count': len(size_df)
        }
    
    return {
        'overall': overall,
        'by_category': by_category,
        'by_coalition_size': by_size
    }

def create_visualizations(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations from results
    
    Args:
        results_df: DataFrame with analysis results
        output_dir: Directory to save visualizations
    """
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Improvement by category
    plt.figure(figsize=(10, 6))
    category_data = []
    
    for category in results_df['task_category'].unique():
        cat_df = results_df[results_df['task_category'] == category]
        category_data.append({
            'Category': category,
            'Improvement': cat_df['improvement'].mean(),
            'Percent': cat_df['percent_improvement'].mean()
        })
    
    cat_df = pd.DataFrame(category_data)
    cat_df = cat_df.sort_values('Improvement', ascending=False)
    
    ax = sns.barplot(x='Category', y='Improvement', data=cat_df)
    ax.bar_label(ax.containers[0], fmt='%.3f')
    plt.title('Average Score Improvement by Task Category')
    plt.ylabel('Score Improvement')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_by_category.png'))
    
    # 2. Improvement by coalition size
    plt.figure(figsize=(10, 6))
    size_data = []
    
    for size in sorted(results_df['coalition_size'].unique()):
        size_df = results_df[results_df['coalition_size'] == size]
        size_data.append({
            'Size': str(size),
            'Improvement': size_df['improvement'].mean(),
            'Percent': size_df['percent_improvement'].mean()
        })
    
    size_df = pd.DataFrame(size_data)
    
    ax = sns.barplot(x='Size', y='Improvement', data=size_df)
    ax.bar_label(ax.containers[0], fmt='%.3f')
    plt.title('Average Score Improvement by Coalition Size')
    plt.xlabel('Coalition Size')
    plt.ylabel('Score Improvement')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_by_size.png'))
    
    # 3. Individual vs Coalition scores (scatter plot)
    plt.figure(figsize=(8, 8))
    plt.scatter(results_df['individual_score'], results_df['coalition_score'], alpha=0.7)
    
    # Add diagonal line (y=x)
    min_val = min(results_df['individual_score'].min(), results_df['coalition_score'].min()) - 0.05
    max_val = max(results_df['individual_score'].max(), results_df['coalition_score'].max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Individual vs Coalition Performance')
    plt.xlabel('Individual Prompt Score')
    plt.ylabel('Coalition Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_vs_coalition.png'))
    
    # 4. Distribution of improvements
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['improvement'], bins=15, kde=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Performance Improvements')
    plt.xlabel('Score Improvement')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_distribution.png'))

def print_summary_report(summary: Dict[str, Any]) -> None:
    """
    Print a summary report of the analysis results
    
    Args:
        summary: Dictionary with summary metrics
    """
    overall = summary['overall']
    
    print("\n" + "=" * 60)
    print("COALITION PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nOverall Results ({overall['total_examples']} examples):")
    print(f"  Average Individual Score:    {overall['avg_individual_score']:.4f}")
    print(f"  Average Coalition Score:     {overall['avg_coalition_score']:.4f}")
    print(f"  Average Improvement:         {overall['avg_improvement']:.4f}")
    print(f"  Average Percent Improvement: {overall['avg_percent_improvement']:.2f}%")
    print(f"  Positive Improvements:       {overall['positive_improvements']} ({overall['positive_improvements']/overall['total_examples']*100:.1f}%)")
    print(f"  Negative Improvements:       {overall['negative_improvements']} ({overall['negative_improvements']/overall['total_examples']*100:.1f}%)")
    
    print("\nResults by Task Category:")
    for category, metrics in summary['by_category'].items():
        print(f"  {category} ({metrics['count']} examples):")
        print(f"    Average Improvement: {metrics['avg_improvement']:.4f} ({metrics['avg_percent_improvement']:.2f}%)")
    
    print("\nResults by Coalition Size:")
    for size, metrics in summary['by_coalition_size'].items():
        print(f"  Size {size} ({metrics['count']} examples):")
        print(f"    Average Improvement: {metrics['avg_improvement']:.4f} ({metrics['avg_percent_improvement']:.2f}%)")
    
    print("\nVisualization files have been saved to the results directory.")
    print("=" * 60)

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Analyze prompt coalition performance")
    parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--examples", type=int, default=5, help="Examples per category")
    parser.add_argument("--sizes", type=str, default="2,3", help="Coalition sizes to test (comma-separated)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Parse coalition sizes
    coalition_sizes = [int(s) for s in args.sizes.split(',')]
    
    # Run analysis
    analyze_coalition_performance(
        prompts_csv=args.csv,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        examples_per_category=args.examples,
        coalition_sizes=coalition_sizes,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()