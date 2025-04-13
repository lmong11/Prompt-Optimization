import pandas as pd
import re
import os
import logging
from typing import Dict, Any, List
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptEvaluator:
    """
    Evaluate prompt performance on task examples
    """
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize the evaluator with API credentials
        
        Args:
            api_key: OpenAI API key (if None, will try to use environment variable)
            model: The LLM model to use for evaluation
        """
        if api_key:
            openai.api_key = api_key
        self.model = model
        
    def run_prompt_evaluation(self, prompt_text: str, task_text: str) -> str:
        """
        Run the LLM with the given prompt and task
        
        Args:
            prompt_text: The system prompt
            task_text: The user task/query
            
        Returns:
            Model response
        """
        try:
            # This is an example using OpenAI's API - adapt as needed for your actual LLM API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": task_text}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM API call: {e}")
            return ""
    
    def evaluate_code_output(self, output: str, task_text: str) -> float:
        """
        Evaluate code outputs by checking if they run and produce expected results
        
        Args:
            output: The code output from the LLM
            task_text: The original task description
            
        Returns:
            Score between 0 and 1
        """
        # In a real implementation, you would extract the code, run it,
        # and check if it produces the expected output
        
        # This is a simplified version that just checks for code presence
        code_pattern = r'```(?:python)?(.*?)```'
        code_matches = re.findall(code_pattern, output, re.DOTALL)
        
        if not code_matches:
            return 0.2  # Low score if no code is present
        
        # Simple heuristics for code quality (simplified version)
        code = code_matches[0]
        
        score = 0.5  # Base score
        
        # Check for basic code quality indicators
        if "def " in code:
            score += 0.1  # Has function definitions
        if "import " in code:
            score += 0.1  # Has imports
        if "return " in code:
            score += 0.1  # Has return statements
        if "if " in code:
            score += 0.1  # Has conditionals
        if "for " in code or "while " in code:
            score += 0.1  # Has loops
            
        # Cap at 1.0
        return min(score, 1.0)
    
    def evaluate_qualitative_output(self, output: str, task_text: str, prompt_text: str) -> float:
        """
        Evaluate qualitative outputs using a judge LLM
        
        Args:
            output: The output from the LLM
            task_text: The original task description
            prompt_text: The system prompt used
            
        Returns:
            Score between 0 and 1
        """
        try:
            # Use a judge LLM to evaluate the output
            judge_prompt = """
            You are an expert evaluator of LLM outputs. Rate the following output 
            on a scale from 0 to 1, where 0 is completely irrelevant or incorrect,
            and 1 is perfect. Be critical and objective in your assessment.
            
            Task description: {task}
            
            System prompt used: {prompt}
            
            LLM output to evaluate: {output}
            
            Provide a score between 0 and 1, and a brief justification.
            Just output the number followed by a colon and your brief justification.
            """
            
            judge_prompt = judge_prompt.format(
                task=task_text,
                prompt=prompt_text,
                output=output
            )
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert LLM output evaluator."},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=200
            )
            
            judge_output = response.choices[0].message.content
            
            # Extract the score (assuming format like "0.8: Good explanation...")
            score_match = re.match(r'^(0\.\d+|[01])(.*)', judge_output)
            if score_match:
                return float(score_match.group(1))
            else:
                # If no clear score found, default to middle score
                return 0.5
            
        except Exception as e:
            logger.error(f"Error in judge LLM API call: {e}")
            return 0.5
    
    def evaluate_pair(self, prompt_text: str, task_text: str, task_category: str) -> Dict[str, Any]:
        """
        Evaluate a prompt-task pair
        
        Args:
            prompt_text: The system prompt
            task_text: The user task/query
            task_category: The category of the task
            
        Returns:
            Dictionary with evaluation results
        """
        # Run the LLM with this prompt-task pair
        output = self.run_prompt_evaluation(prompt_text, task_text)
        
        # Evaluate based on task category
        if task_category.startswith('CG'):  # Code Generation
            score = self.evaluate_code_output(output, task_text)
        else:  # Qualitative evaluation
            score = self.evaluate_qualitative_output(output, task_text, prompt_text)
            
        return {
            "output": output,
            "score": score
        }
    
    def batch_evaluate(self, evaluation_pairs: pd.DataFrame, output_path: str) -> pd.DataFrame:
        """
        Evaluate a batch of prompt-task pairs
        
        Args:
            evaluation_pairs: DataFrame with prompt-task pairs
            output_path: Path to save the evaluation results
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for i, row in evaluation_pairs.iterrows():
            logger.info(f"Evaluating pair {i+1}/{len(evaluation_pairs)}: {row['prompt_id']} + {row['task_id']}")
            
            try:
                eval_result = self.evaluate_pair(
                    row['prompt_text'], 
                    row['task_text'],
                    row['task_category']
                )
                
                results.append({
                    "prompt_id": row['prompt_id'],
                    "task_id": row['task_id'],
                    "output": eval_result["output"],
                    "score": eval_result["score"],
                    "prompt_category": row['prompt_category'],
                    "task_category": row['task_category']
                })
                
                # Periodically save results
                if (i + 1) % 10 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(f"{output_path}_temp.csv", index=False)
                    
            except Exception as e:
                logger.error(f"Error evaluating pair: {e}")
                
                # Add an entry with error
                results.append({
                    "prompt_id": row['prompt_id'],
                    "task_id": row['task_id'],
                    "output": f"ERROR: {str(e)}",
                    "score": 0.0,
                    "prompt_category": row['prompt_category'],
                    "task_category": row['task_category']
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        
        return results_df