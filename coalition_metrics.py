#!/usr/bin/env python3
"""
Coalition Metrics Evaluator

This script evaluates and visualizes the performance improvements 
from using prompt coalitions versus individual prompts.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
import json

# Import from the main project
from data_preparation import PromptDataProcessor
from prompt_selection_system import PromptSelectionSystem
from prompt_evaluator import PromptEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("coalition_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CoalitionMetricsEvaluator:
    """
    Evaluates and compares metrics between individual prompts and prompt coalitions
    """
    def __init__(
        self,
        prompts_csv: str,
        data_dir: str = "data",
        model_dir: str = "models",
        results_dir: str = "results",
        openai_api_key: str = None
    ):
        """
        Initialize the evaluator
        
        Args:
            prompts_csv: Path to CSV file with system prompts
            data_dir: Directory with processed data
            model_dir: Directory with trained models
            results_dir: Directory to store evaluation results
            openai_api_key: OpenAI API key for evaluation (optional)
        """
        self.prompts_csv = prompts_csv
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.openai_api_key = openai_api_key
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize prompt selection system
        self.system = PromptSelectionSystem(
            prompts_csv=prompts_csv,
            data_dir=data_dir,
            model_dir=model_dir,
            openai_api_key=openai_api_key
        )
        
        # Initialize evaluator if API key is provided
        if openai_api_key:
            self.evaluator = PromptEvaluator(api_key=openai_api_key)
        else:
            self.evaluator = None
            
    def evaluate_test_set(
        self, 
        test_size: int = 20, 
        max_coalition_size: int = 3,
        use_api: bool = False
    ) -> pd.DataFrame:
        """
        Evaluate individual prompts versus coalitions on a test set
        
        Args:
            test_size: Number of test examples to evaluate
            max_coalition_size: Maximum size of prompt coalitions
            use_api: Whether to use real API evaluation (requires API key)
            
        Returns:
            DataFrame with evaluation results
        """
        # Load test data
        try:
            test_df = pd.read_csv(f"{self.data_dir}/test_data.csv")
            logger.info(f"Loaded test data with {len(test_df)} examples")
        except FileNotFoundError:
            logger.error(f"Test data not found. Run process_data first.")
            raise
        
        # Sample test examples
        test_sample = test_df.sample(min(test_size, len(test_df)), random_state=42)
        
        results = []
        
        for i, row in test_sample.iterrows():
            task_text = row['task_text']
            task_category = row['task_category']
            task_id = row['task_id']
            
            logger.info(f"Evaluating example {i+1}/{len(test_sample)}: {task_text[:50]}...")
            
            # Select individual prompt
            individual_result = self.system.select_optimal_prompt(
                task_text=task_text,
                task_category=task_category,
                coalition_mode=False
            )
            
            # Select prompt coalition
            coalition_result = self.system.select_optimal_prompt(
                task_text=task_text,
                task_category=task_category,
                coalition_mode=True,
                max_coalition_size=max_coalition_size
            )
            
            # Record base results
            result = {
                'task_id': task_id,
                'task_text': task_text,
                'task_category': task_category,
                'individual_prompt_id': individual_result['prompt_id'],
                'individual_prompt_name': individual_result['prompt_name'],
                'coalition_ids': ','.join([p['id'] for p in coalition_result['selected_prompts']]),
                'coalition_names': ','.join([p['name'] for p in coalition_result['selected_prompts']]),
                'coalition_size': len(coalition_result['selected_prompts'])
            }
            
            # If using API evaluation, get actual performance metrics
            if use_api and self.evaluator:
                # Evaluate individual prompt
                individual_eval = self.evaluator.evaluate_pair(
                    individual_result['prompt_text'],
                    task_text,
                    task_category
                )
                
                # Evaluate coalition prompt
                coalition_eval = self.evaluator.evaluate_pair(
                    coalition_result['combined_prompt'],
                    task_text,
                    task_category
                )
                
                # Add evaluation scores
                result['individual_score'] = individual_eval['score']
                result['coalition_score'] = coalition_eval['score']
                result['improvement'] = coalition_eval['score'] - individual_eval['score']
                result['percent_improvement'] = (
                    (coalition_eval['score'] - individual_eval['score']) / 
                    max(individual_eval['score'], 0.001) * 100
                )
            else:
                # Predict scores using the utility model
                individual_data = pd.DataFrame([{
                    'prompt_text': individual_result['prompt_text'],
                    'prompt_category': individual_result['prompt_category'] if 'prompt_category' in individual_result else task_category,
                    'task_text': task_text,
                    'task_category': task_category
                }])
                
                coalition_data = pd.DataFrame([{
                    'prompt_text': coalition_result['combined_prompt'],
                    'prompt_category': '+'.join([p['category'] for p in coalition_result['selected_prompts']]),
                    'task_text': task_text,
                    'task_category': task_category
                }])
                
                # Get predicted scores
                individual_score = self.system.utility_model.predict(individual_data)[0]
                coalition_score = self.system.utility_model.predict(coalition_data)[0]
                
                # Add predicted scores
                result['individual_score'] = individual_score
                result['coalition_score'] = coalition_score
                result['improvement'] = coalition_score - individual_score
                result['percent_improvement'] = (
                    (coalition_score - individual_score) / 
                    max(individual_score, 0.001) * 100
                )
            
            results.append(result)
            
            # Print progress
            logger.info(
                f"Task {i+1}: Individual={result['individual_score']:.3f}, "
                f"Coalition={result['coalition_score']:.3f}, "
                f"Improvement={result['improvement']:.3f} ({result['percent_improvement']:.1f}%)"
            )
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(self.results_dir, "coalition_metrics.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze evaluation results and compute summary metrics
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Dictionary with summary metrics
        """
        # Calculate overall metrics
        overall_metrics = {
            'avg_individual_score': results_df['individual_score'].mean(),
            'avg_coalition_score': results_df['coalition_score'].mean(),
            'avg_improvement': results_df['improvement'].mean(),
            'avg_percent_improvement': results_df['percent_improvement'].mean(),
            'median_improvement': results_df['improvement'].median(),
            'max_improvement': results_df['improvement'].max(),
            'min_improvement': results_df['improvement'].min(),
            'positive_improvements': (results_df['improvement'] > 0).sum(),
            'negative_improvements': (results_df['improvement'] < 0).sum(),
            'total_examples': len(results_df)
        }
        
        # Calculate metrics by task category
        category_metrics = {}
        for category in results_df['task_category'].unique():
            cat_df = results_df[results_df['task_category'] == category]
            category_metrics[category] = {
                'avg_individual_score': cat_df['individual_score'].mean(),
                'avg_coalition_score': cat_df['coalition_score'].mean(),
                'avg_improvement': cat_df['improvement'].mean(),
                'avg_percent_improvement': cat_df['percent_improvement'].mean(),
                'count': len(cat_df)
            }
        
        # Calculate metrics by coalition size
        size_metrics = {}
        for size in results_df['coalition_size'].unique():
            size_df = results_df[results_df['coalition_size'] == size]
            size_metrics[str(size)] = {
                'avg_individual_score': size_df['individual_score'].mean(),
                'avg_coalition_score': size_df['coalition_score'].mean(),
                'avg_improvement': size_df['improvement'].mean(),
                'avg_percent_improvement': size_df['percent_improvement'].mean(),
                'count': len(size_df)
            }
        
        # Combined metrics
        analysis = {
            'overall': overall_metrics,
            'by_category': category_metrics,
            'by_coalition_size': size_metrics
        }
        
        # Save analysis
        output_path = os.path.join(self.results_dir, "coalition_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {output_path}")
        
        return analysis
    
    def visualize_results(self, results_df: pd.DataFrame) -> None:
        """
        Create visualizations of the evaluation results
        
        Args:
            results_df: DataFrame with evaluation results
        """
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Overall score comparison (boxplot)
        plt.figure(figsize=(10, 6))
        scores_df = pd.DataFrame({
            'Individual Prompt': results_df['individual_score'],
            'Prompt Coalition': results_df['coalition_score']
        })
        sns.boxplot(data=scores_df)
        plt.title('Performance Comparison: Individual vs Coalition')
        plt.ylabel('Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'score_comparison_boxplot.png'))
        
        # 2. Improvement distribution (histogram)
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['improvement'], bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Distribution of Score Improvements')
        plt.xlabel('Score Improvement (Coalition - Individual)')
        plt.ylabel('Count')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'improvement_histogram.png'))
        
        # 3. Performance by task category
        plt.figure(figsize=(12, 6))
        category_data = []
        
        for category in results_df['task_category'].unique():
            cat_df = results_df[results_df['task_category'] == category]
            category_data.append({
                'Category': category,
                'Individual': cat_df['individual_score'].mean(),
                'Coalition': cat_df['coalition_score'].mean()
            })
        
        cat_df = pd.DataFrame(category_data)
        cat_df = pd.melt(cat_df, id_vars=['Category'], var_name='Method', value_name='Score')
        
        sns.barplot(x='Category', y='Score', hue='Method', data=cat_df)
        plt.title('Average Performance by Task Category')
        plt.xlabel('Task Category')
        plt.ylabel('Average Score')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_by_category.png'))
        
        # 4. Performance by coalition size
        plt.figure(figsize=(10, 6))
        size_data = []
        
        for size in sorted(results_df['coalition_size'].unique()):
            size_df = results_df[results_df['coalition_size'] == size]
            size_data.append({
                'Size': size,
                'Individual': size_df['individual_score'].mean(),
                'Coalition': size_df['coalition_score'].mean()
            })
        
        size_df = pd.DataFrame(size_data)
        size_df = pd.melt(size_df, id_vars=['Size'], var_name='Method', value_name='Score')
        
        sns.barplot(x='Size', y='Score', hue='Method', data=size_df)
        plt.title('Average Performance by Coalition Size')
        plt.xlabel('Coalition Size')
        plt.ylabel('Average Score')
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_by_size.png'))
        
        # 5. Scatter plot of individual vs coalition scores
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
        plt.savefig(os.path.join(self.results_dir, 'individual_vs_coalition.png'))
        
        logger.info(f"Visualizations saved to {self.results_dir}")
    
    def run_evaluation(
        self, 
        test_size: int = 20, 
        max_coalition_size: int = 3,
        use_api: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline
        
        Args:
            test_size: Number of test examples to evaluate
            max_coalition_size: Maximum size of prompt coalitions
            use_api: Whether to use real API evaluation
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting coalition metrics evaluation on {test_size} examples")
        
        # Evaluate test set
        results_df = self.evaluate_test_set(
            test_size=test_size,
            max_coalition_size=max_coalition_size,
            use_api=use_api
        )
        
        # Analyze results
        analysis = self.analyze_results(results_df)
        
        # Create visualizations
        self.visualize_results(results_df)
        
        # Print summary report
        self._print_summary(analysis)
        
        logger.info("Coalition metrics evaluation completed successfully")
        return analysis
    
    def _print_summary(self, analysis: Dict[str, Any]) -> None:
        """
        Print a summary of the analysis results
        
        Args:
            analysis: Dictionary with analysis results
        """
        overall = analysis['overall']
        
        print("\n" + "=" * 60)
        print("COALITION METRICS EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Results ({overall['total_examples']} examples):")
        print(f"  Average Individual Score:    {overall['avg_individual_score']:.4f}")
        print(f"  Average Coalition Score:     {overall['avg_coalition_score']:.4f}")
        print(f"  Average Improvement:         {overall['avg_improvement']:.4f}")
        print(f"  Average Percent Improvement: {overall['avg_percent_improvement']:.2f}%")
        print(f"  Positive Improvements:       {overall['positive_improvements']} ({overall['positive_improvements']/overall['total_examples']*100:.1f}%)")
        print(f"  Negative Improvements:       {overall['negative_improvements']} ({overall['negative_improvements']/overall['total_examples']*100:.1f}%)")
        
        print("\nResults by Task Category:")
        for category, metrics in analysis['by_category'].items():
            print(f"  {category} ({metrics['count']} examples):")
            print(f"    Average Improvement: {metrics['avg_improvement']:.4f} ({metrics['avg_percent_improvement']:.2f}%)")
        
        print("\nResults by Coalition Size:")
        for size, metrics in analysis['by_coalition_size'].items():
            print(f"  Size {size} ({metrics['count']} examples):")
            print(f"    Average Improvement: {metrics['avg_improvement']:.4f} ({metrics['avg_percent_improvement']:.2f}%)")
        
        print("\nVisualization files have been saved to the results directory.")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Coalition Metrics Evaluator")
    parser.add_argument("--csv", type=str, default="System_Prompt.csv", help="Path to prompts CSV file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Model directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--test_size", type=int, default=20, help="Number of test examples to evaluate")
    parser.add_argument("--max_coalition", type=int, default=3, help="Maximum coalition size")
    parser.add_argument("--use_api", action="store_true", help="Use real API evaluation")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (required if --use_api is used)")
    
    args = parser.parse_args()
    
    if args.use_api and not args.api_key:
        parser.error("--api_key is required when --use_api is specified")
    
    # Create evaluator
    evaluator = CoalitionMetricsEvaluator(
        prompts_csv=args.csv,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        openai_api_key=args.api_key
    )
    
    # Run evaluation
    evaluator.run_evaluation(
        test_size=args.test_size,
        max_coalition_size=args.max_coalition,
        use_api=args.use_api
    )

if __name__ == "__main__":
    main()
