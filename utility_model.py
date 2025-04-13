import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Tuple, Union, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UtilityFunctionModel:
    """
    Model to predict utility of a prompt for a given task
    """
    def __init__(self, 
                 embedding_model_name: str = 'all-MiniLM-L6-v2', 
                 model_dir: str = 'models'):
        """
        Initialize the utility function model
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
            model_dir: Directory to store trained models
        """
        self.model_dir = model_dir
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.utility_model = None
        self.category_encoder = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_features(self, 
                         data: pd.DataFrame, 
                         fit_encoder: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for model training/prediction
        
        Args:
            data: DataFrame with prompt-task pairs and scores
            fit_encoder: Whether to fit the category encoder (for training) or use existing (for inference)
            
        Returns:
            Tuple of features array and feature names
        """
        # Get prompt and task embeddings
        prompt_embeddings = self.embedding_model.encode(data['prompt_text'].tolist())
        task_embeddings = self.embedding_model.encode(data['task_text'].tolist())
        
        # Prepare categorical features
        categorical_features = data[['prompt_category', 'task_category']]
        
        if fit_encoder or self.category_encoder is None:
            self.category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_encoded = self.category_encoder.fit_transform(categorical_features)
        else:
            categorical_encoded = self.category_encoder.transform(categorical_features)
        
        # Combine all features
        features = np.hstack([
            prompt_embeddings,  # Prompt embeddings
            task_embeddings,    # Task embeddings
            categorical_encoded  # Categorical features
        ])
        
        # Create feature names for interpretability
        feature_names = (
            [f'prompt_emb_{i}' for i in range(self.embedding_dim)] +
            [f'task_emb_{i}' for i in range(self.embedding_dim)] +
            self.category_encoder.get_feature_names_out(['prompt_category', 'task_category']).tolist()
        )
        
        return features, feature_names
    
    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train the utility function model
        
        Args:
            train_data: DataFrame with prompt-task pairs and scores for training
            validation_data: Optional DataFrame for validation
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features and target
        X_train, feature_names = self.prepare_features(train_data, fit_encoder=True)
        y_train = train_data['score'].values
        
        # Initialize and train the model
        logger.info(f"Training utility model on {len(train_data)} examples with {X_train.shape[1]} features")
        self.utility_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.utility_model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_preds = self.utility_model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)
        
        metrics = {
            'train_mse': train_mse,
            'train_r2': train_r2
        }
        
        # Calculate validation metrics if validation data is provided
        if validation_data is not None:
            X_val, _ = self.prepare_features(validation_data)
            y_val = validation_data['score'].values
            
            val_preds = self.utility_model.predict(X_val)
            val_mse = mean_squared_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            
            metrics.update({
                'val_mse': val_mse,
                'val_r2': val_r2
            })
        
        # Save the model
        self._save_model()
        
        logger.info(f"Model trained successfully. Metrics: {metrics}")
        return metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict utility scores for prompt-task pairs
        
        Args:
            data: DataFrame with prompt-task pairs
            
        Returns:
            Array of predicted utility scores
        """
        if self.utility_model is None:
            self._load_model()
            
        X, _ = self.prepare_features(data)
        return self.utility_model.predict(X)
    
    def _save_model(self):
        """Save the trained model and encoder"""
        model_path = os.path.join(self.model_dir, 'utility_model.joblib')
        encoder_path = os.path.join(self.model_dir, 'category_encoder.joblib')
        
        joblib.dump(self.utility_model, model_path)
        joblib.dump(self.category_encoder, encoder_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def _load_model(self):
        """Load a saved model and encoder"""
        model_path = os.path.join(self.model_dir, 'utility_model.joblib')
        encoder_path = os.path.join(self.model_dir, 'category_encoder.joblib')
        
        try:
            self.utility_model = joblib.load(model_path)
            self.category_encoder = joblib.load(encoder_path)
            logger.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            raise ValueError("No trained model found. Please train the model first.")


class CoalitionGamePromptSelector:
    """
    Prompt selector based on coalition game theory
    """
    def __init__(self, utility_model: UtilityFunctionModel):
        """
        Initialize the coalition game prompt selector
        
        Args:
            utility_model: Trained utility function model
        """
        self.utility_model = utility_model
    
    def compute_shapley_values(self, 
                               prompts: List[Dict[str, Any]], 
                               task: Dict[str, Any],
                               max_coalition_size: int = 3) -> Dict[str, float]:
        """
        Compute the Shapley value for each prompt given a task
        
        Args:
            prompts: List of prompt dictionaries with id, text, and category
            task: Task dictionary with text and category
            max_coalition_size: Maximum number of prompts to include in a coalition
            
        Returns:
            Dictionary mapping prompt IDs to their Shapley values
        """
        n = len(prompts)
        if n > 7:
            logger.warning(f"Computing exact Shapley values for {n} prompts may be slow. Consider using approximation.")
        
        shapley_values = {}
        
        # Create a task DataFrame with repeated rows for each prompt
        task_df = pd.DataFrame([{
            'task_text': task['text'],
            'task_category': task['category']
        }] * n)
        
        # For each prompt, compute individual utility
        individual_utilities = {}
        prompt_df = pd.DataFrame([{
            'prompt_id': p['id'],
            'prompt_text': p['text'],
            'prompt_category': p['category'],
            'task_text': task['text'],
            'task_category': task['category']
        } for p in prompts])
        
        # Predict utilities for all prompts
        utilities = self.utility_model.predict(prompt_df)
        
        for i, p in enumerate(prompts):
            individual_utilities[p['id']] = utilities[i]
        
        # Compute exact Shapley values
        # Note: This implementation can be slow for many prompts
        # In practice, you may want to use approximation methods
        for i, prompt in enumerate(prompts):
            prompt_id = prompt['id']
            marginal_contributions = []
            
            # Use binary representation to enumerate all subsets
            # Only consider coalitions up to max_coalition_size
            for j in range(2**n):
                # Convert j to binary, representing subset inclusion
                subset = [k for k in range(n) if (j >> k) & 1]
                
                # Skip if subset contains the current prompt or is too large
                if i in subset or len(subset) >= max_coalition_size:
                    continue
                
                # Calculate utility without the current prompt
                subset_utility = self._get_coalition_utility(
                    [prompts[k] for k in subset], 
                    task
                )
                
                # Calculate utility with the current prompt added
                subset_with_i = subset + [i]
                subset_with_i_utility = self._get_coalition_utility(
                    [prompts[k] for k in subset_with_i], 
                    task
                )
                
                # Calculate marginal contribution
                marginal_contribution = subset_with_i_utility - subset_utility
                marginal_contributions.append(marginal_contribution)
            
            # Shapley value is the average of marginal contributions
            if marginal_contributions:
                shapley_values[prompt_id] = sum(marginal_contributions) / len(marginal_contributions)
            else:
                shapley_values[prompt_id] = individual_utilities[prompt_id]
        
        return shapley_values
    
    def _get_coalition_utility(self, prompt_coalition: List[Dict[str, Any]], task: Dict[str, Any]) -> float:
        """
        Calculate the utility of a coalition of prompts for a given task
        
        Args:
            prompt_coalition: List of prompts in the coalition
            task: Task to calculate utility for
            
        Returns:
            Utility score of the coalition
        """
        if not prompt_coalition:
            return 0.0
        
        # For single prompt, use the utility model directly
        if len(prompt_coalition) == 1:
            prompt_df = pd.DataFrame([{
                'prompt_id': prompt_coalition[0]['id'],
                'prompt_text': prompt_coalition[0]['text'],
                'prompt_category': prompt_coalition[0]['category'],
                'task_text': task['text'],
                'task_category': task['category']
            }])
            
            return self.utility_model.predict(prompt_df)[0]
        
        # For multiple prompts, combine them and calculate utility
        combined_prompt_text = " ".join([p['text'] for p in prompt_coalition])
        combined_prompt_category = "+".join([p['category'] for p in prompt_coalition])
        
        combined_prompt_df = pd.DataFrame([{
            'prompt_id': "+".join([p['id'] for p in prompt_coalition]),
            'prompt_text': combined_prompt_text,
            'prompt_category': combined_prompt_category,
            'task_text': task['text'],
            'task_category': task['category']
        }])
        
        # Get utility of combined prompt
        combined_utility = self.utility_model.predict(combined_prompt_df)[0]
        
        # Apply a synergy bonus if the coalition includes diverse categories
        categories = set([p['category'] for p in prompt_coalition])
        if len(categories) > 1:
            # Simple synergy bonus - could be more sophisticated in a real implementation
            synergy_bonus = 0.05 * (len(categories) - 1)
            combined_utility *= (1 + synergy_bonus)
        
        return combined_utility
    
    def find_optimal_coalition(self, 
                               prompts: List[Dict[str, Any]], 
                               task: Dict[str, Any],
                               max_coalition_size: int = 3) -> List[Dict[str, Any]]:
        """
        Find the optimal coalition of prompts for a given task
        
        Args:
            prompts: List of prompt dictionaries
            task: Task dictionary
            max_coalition_size: Maximum size of the coalition
            
        Returns:
            List of prompts in the optimal coalition
        """
        # Compute Shapley values for each prompt
        shapley_values = self.compute_shapley_values(prompts, task, max_coalition_size)
        
        # Sort prompts by Shapley value
        sorted_prompts = sorted(
            [(prompt, shapley_values.get(prompt['id'], 0)) 
             for prompt in prompts],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take the top prompts up to max_coalition_size
        top_prompts = [p[0] for p in sorted_prompts[:max_coalition_size]]
        
        # Verify if this coalition is better than smaller ones
        best_coalition = [top_prompts[0]]
        best_utility = self._get_coalition_utility(best_coalition, task)
        
        for i in range(1, len(top_prompts)):
            candidate_coalition = best_coalition + [top_prompts[i]]
            candidate_utility = self._get_coalition_utility(candidate_coalition, task)
            
            # Only add prompt if it improves utility
            if candidate_utility > best_utility:
                best_coalition = candidate_coalition
                best_utility = candidate_utility
            
        return best_coalition
    
    def select_best_prompt(self,
                          prompts: List[Dict[str, Any]],
                          task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best single prompt for a given task
        
        Args:
            prompts: List of prompt dictionaries
            task: Task dictionary
            
        Returns:
            Best prompt dictionary
        """
        # Create DataFrame for prediction
        prompt_df = pd.DataFrame([{
            'prompt_id': p['id'],
            'prompt_text': p['text'],
            'prompt_category': p['category'],
            'task_text': task['text'],
            'task_category': task['category']
        } for p in prompts])
        
        # Predict utilities
        utilities = self.utility_model.predict(prompt_df)
        
        # Find prompt with highest utility
        best_idx = np.argmax(utilities)
        
        return prompts[best_idx]


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load evaluation results (assuming they exist)
    try:
        results_df = pd.read_csv("data/evaluation_results.csv")
    
        # Train utility model
        utility_model = UtilityFunctionModel()
        train_df, test_df = train_test_split(results_df, test_size=0.2, random_state=42)
        metrics = utility_model.train(train_df, test_df)
        
        print(f"Utility model trained with metrics: {metrics}")
        
        # Test prompt selection
        prompts = [
            {
                'id': 'CG-1',
                'text': 'You are an algorithmic reasoning expert...',
                'category': 'CG'
            },
            {
                'id': 'DA-2',
                'text': 'You are a data analysis expert...',
                'category': 'DA'
            }
        ]
        
        task = {
            'text': 'Implement a function to find the kth largest element in an unsorted array',
            'category': 'CG'
        }
        
        selector = CoalitionGamePromptSelector(utility_model)
        best_prompt = selector.select_best_prompt(prompts, task)
        print(f"Best prompt: {best_prompt['id']}")
        
        optimal_coalition = selector.find_optimal_coalition(prompts, task)
        print(f"Optimal coalition: {[p['id'] for p in optimal_coalition]}")
        
    except FileNotFoundError:
        print("No evaluation results found. Run the evaluator first.")