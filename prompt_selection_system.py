import pandas as pd
import numpy as np
import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import openai

# Import local modules
from data_preparation import PromptDataProcessor, RealDatasetManager
from prompt_evaluator import PromptEvaluator
from utility_model import UtilityFunctionModel, CoalitionGamePromptSelector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptSelectionSystem:
    """
    End-to-end system for prompt selection and evaluation
    """
    def __init__(self, 
                 prompts_csv: str,
                 data_dir: str = "data",
                 model_dir: str = "models",
                 openai_api_key: Optional[str] = None,
                 llm_model: str = "gpt-4"):
        """
        Initialize the prompt selection system
        
        Args:
            prompts_csv: Path to CSV file with system prompts
            data_dir: Directory for storing processed data
            model_dir: Directory for storing trained models
            openai_api_key: OpenAI API key for evaluation (optional)
            llm_model: LLM model to use for evaluation
        """
        self.prompts_csv = prompts_csv
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Initialize data processor
        self.data_processor = PromptDataProcessor(prompts_csv, output_dir=data_dir)
        
        # Initialize utility model
        self.utility_model = UtilityFunctionModel(model_dir=model_dir)
        
        # Initialize prompt selector
        self.prompt_selector = CoalitionGamePromptSelector(self.utility_model)
        
        # Initialize evaluator if API key is provided
        if openai_api_key:
            openai.api_key = openai_api_key
            self.evaluator = PromptEvaluator(model=llm_model)
        else:
            self.evaluator = None
            
        # Initialize embedding model for task categorization
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training and evaluation
        
        Returns:
            Tuple of train and test DataFrames
        """
        logger.info("Loading prompts from CSV file")
        self.data_processor.load_prompts()
        
        logger.info("Generating task examples from real datasets")
        self.data_processor.generate_task_examples()
        
        logger.info("Preparing training and test data")
        train_df, test_df = self.data_processor.prepare_training_data()
        
        return train_df, test_df
    
    def evaluate_prompts(self, eval_data: pd.DataFrame) -> pd.DataFrame:
        """
        Run evaluation on prompt-task pairs
        
        Args:
            eval_data: DataFrame with prompt-task pairs to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized. Please provide an OpenAI API key.")
        
        output_path = os.path.join(self.data_dir, "evaluation_results.csv")
        
        logger.info(f"Evaluating {len(eval_data)} prompt-task pairs")
        results_df = self.evaluator.batch_evaluate(eval_data, output_path)
        
        return results_df
    
    def train_utility_model(self, results_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the utility function model
        
        Args:
            results_data: DataFrame with evaluation results
            
        Returns:
            Dictionary with training metrics
        """
        # Split into train and validation sets
        train_data, val_data = train_test_split(
            results_data, test_size=0.2, random_state=42,
            stratify=results_data[['prompt_category', 'task_category']]
        )
        
        logger.info(f"Training utility model on {len(train_data)} examples")
        metrics = self.utility_model.train(train_data, val_data)
        
        return metrics
    
    def select_optimal_prompt(self, 
                             task_text: str, 
                             task_category: Optional[str] = None,
                             coalition_mode: bool = False,
                             max_coalition_size: int = 3) -> Dict[str, Any]:
        """
        Select the optimal prompt or prompt coalition for a given task
        
        Args:
            task_text: The task/query text
            task_category: Optional task category (will be inferred if not provided)
            coalition_mode: Whether to use coalition game approach
            max_coalition_size: Maximum size of prompt coalition
            
        Returns:
            Dictionary with selected prompt(s) and metadata
        """
        # Load prompts
        if not hasattr(self.data_processor, 'prompts') or self.data_processor.prompts is None:
            self.data_processor.load_prompts()
        
        # Infer task category if not provided
        if task_category is None:
            task_category = self._infer_task_category(task_text)
        
        # Prepare prompt dictionaries
        prompts = []
        for _, row in self.data_processor.prompts.iterrows():
            prompts.append({
                'id': row['Prompt ID'],
                'text': row['Clean Prompt Text'],
                'category': row['Task Category'],
                'name': row['Prompt Name']
            })
        
        # Task dictionary
        task = {
            'text': task_text,
            'category': task_category
        }
        
        # Select optimal prompts
        if coalition_mode:
            logger.info(f"Finding optimal prompt coalition for task: {task_text[:50]}...")
            selected_prompts = self.prompt_selector.find_optimal_coalition(
                prompts, task, max_coalition_size
            )
            
            # Combined prompt text
            combined_text = "\n\n".join([p['text'] for p in selected_prompts])
            
            return {
                'mode': 'coalition',
                'selected_prompts': selected_prompts,
                'prompt_ids': [p['id'] for p in selected_prompts],
                'prompt_names': [p['name'] for p in selected_prompts],
                'combined_prompt': combined_text,
                'task_category': task_category
            }
        else:
            logger.info(f"Selecting best individual prompt for task: {task_text[:50]}...")
            best_prompt = self.prompt_selector.select_best_prompt(prompts, task)
            
            return {
                'mode': 'individual',
                'selected_prompt': best_prompt,
                'prompt_id': best_prompt['id'],
                'prompt_name': best_prompt['name'],
                'prompt_text': best_prompt['text'],
                'task_category': task_category
            }
    
    def _infer_task_category(self, task_text: str) -> str:
        """
        Infer the category of a task based on its text
        
        Args:
            task_text: The task text
            
        Returns:
            Inferred task category
        """
        # Get unique categories from prompts
        if not hasattr(self.data_processor, 'prompts') or self.data_processor.prompts is None:
            self.data_processor.load_prompts()
        
        categories = self.data_processor.prompts['Task Category'].unique()
        
        # Get example tasks for each category
        category_examples = {}
        for cat in categories:
            cat_prompts = self.data_processor.prompts[
                self.data_processor.prompts['Task Category'] == cat
            ]
            examples = []
            for _, row in cat_prompts.iterrows():
                if pd.notna(row['Example Inputs']):
                    examples.append(row['Example Inputs'])
            
            if examples:
                category_examples[cat] = examples
        
        # Embed the task text
        task_embedding = self.embedding_model.encode([task_text])[0]
        
        # Embed examples for each category and compute similarity
        category_scores = {}
        for cat, examples in category_examples.items():
            # Clean examples and embed
            clean_examples = []
            for ex in examples:
                if isinstance(ex, str):
                    ex_clean = ex.strip('"\'')
                    if ex_clean:
                        clean_examples.append(ex_clean)
            
            if clean_examples:
                example_embeddings = self.embedding_model.encode(clean_examples)
                
                # Compute cosine similarities
                similarities = np.dot(example_embeddings, task_embedding) / (
                    np.linalg.norm(example_embeddings, axis=1) * np.linalg.norm(task_embedding)
                )
                
                # Use max similarity as category score
                category_scores[cat] = np.max(similarities)
        
        # Return category with highest score, or default to 'CG' if no scores
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'CG'  # Default to code generation if no scores
    
    def run_end_to_end_pipeline(self, 
                              sample_size: int = 100,
                              eval_mode: bool = False) -> None:
        """
        Run the complete end-to-end pipeline: data preparation, evaluation,
        model training, and validation
        
        Args:
            sample_size: Number of samples to use for evaluation (if eval_mode is True)
            eval_mode: Whether to run evaluation (requires API key)
        """
        # Step 1: Prepare data
        logger.info("Step 1: Preparing data")
        train_df, test_df = self.prepare_data()
        
        # Step 2: Evaluate prompts (if in eval mode)
        if eval_mode:
            if self.evaluator is None:
                raise ValueError("Evaluator not initialized. Please provide an OpenAI API key.")
            
            logger.info(f"Step 2: Evaluating prompts on {sample_size} examples")
            # Take a sample for evaluation to reduce API costs
            eval_sample = train_df.sample(min(sample_size, len(train_df)), random_state=42)
            results_df = self.evaluate_prompts(eval_sample)
        else:
            # Try to load existing results
            try:
                results_path = os.path.join(self.data_dir, "evaluation_results.csv")
                results_df = pd.read_csv(results_path)
                logger.info(f"Loaded existing evaluation results from {results_path}")
            except FileNotFoundError:
                logger.warning("No evaluation results found and eval_mode is False. "
                             "Using synthetic scores for demonstration.")
                
                # Create synthetic scores for demonstration
                results_df = train_df.sample(min(sample_size, len(train_df)), random_state=42).copy()
                
                # Assign higher scores when prompt and task categories match
                results_df['score'] = np.random.uniform(0.3, 0.7, size=len(results_df))
                mask = results_df['prompt_category'] == results_df['task_category']
                results_df.loc[mask, 'score'] = np.random.uniform(0.7, 0.95, size=mask.sum())
                
                # Save synthetic results
                results_df.to_csv(os.path.join(self.data_dir, "synthetic_results.csv"), index=False)
        
        # Step 3: Train utility model
        logger.info("Step 3: Training utility model")
        metrics = self.train_utility_model(results_df)
        logger.info(f"Utility model trained with metrics: {metrics}")
        
        # Step 4: Validate on test set
        logger.info("Step 4: Validating prompt selection on test examples")
        
        # Take a few examples for validation
        validation_examples = test_df.sample(min(5, len(test_df)), random_state=42)
        
        for i, row in validation_examples.iterrows():
            task_text = row['task_text']
            task_category = row['task_category']
            
            # Select optimal prompt (individual)
            selection_result = self.select_optimal_prompt(
                task_text, task_category, coalition_mode=False
            )
            logger.info(f"Example {i+1}: Selected prompt {selection_result['prompt_id']} "
                     f"({selection_result['prompt_name']}) for task: {task_text[:50]}...")
            
            # Select optimal prompt coalition
            coalition_result = self.select_optimal_prompt(
                task_text, task_category, coalition_mode=True, max_coalition_size=2
            )
            logger.info(f"Example {i+1}: Selected prompt coalition "
                     f"{coalition_result['prompt_ids']} for task: {task_text[:50]}...")
        
        logger.info("End-to-end pipeline completed successfully")