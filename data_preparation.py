import pandas as pd
import numpy as np
import os
import json
import re
import requests
import zipfile
import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional
import logging
import huggingface_hub
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetManager:
    """Manager for downloading and processing datasets from the table"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the dataset manager
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = data_dir
        self.datasets_dir = os.path.join(data_dir, "datasets")
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Dataset definitions mapping to the table
        self.datasets = {
            "Code Generation": [
                {
                    "name": "HumanEval",
                    "source": "openai_humaneval",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://github.com/openai/human-eval",
                    "task_formatter": self._format_code_generation_task
                },
                {
                    "name": "MBPP",
                    "source": "mbpp",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://github.com/google-research/google-research/tree/master/mbpp",
                    "task_formatter": self._format_mbpp_task
                },
                {
                    "name": "CodeContests",
                    "source": "hendrycks/apps",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://github.com/hendrycks/apps",
                    "task_formatter": self._format_code_contests_task
                }
            ],
            "Data Analysis": [
                {
                    "name": "Titanic",
                    "source": "csv:https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
                    "load_method": "csv",
                    "url": "https://www.kaggle.com/c/titanic",
                    "task_formatter": self._format_titanic_task
                },
                {
                    "name": "Housing",
                    "source": "csv:https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv", 
                    "load_method": "csv",
                    "url": "https://www.kaggle.com/c/boston-housing",
                    "task_formatter": self._format_housing_task
                },
                {
                    "name": "FiveThirtyEight",
                    "source": "json:https://raw.githubusercontent.com/fivethirtyeight/data/master/polls/polls.csv",
                    "load_method": "csv",
                    "url": "https://data.fivethirtyeight.com",
                    "task_formatter": self._format_538_task
                }
            ],
            "Text Summarization": [
                {
                    "name": "CNN/Daily Mail",
                    "source": "cnn_dailymail",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://huggingface.co/datasets/cnn_dailymail",
                    "task_formatter": self._format_cnn_dailymail_task
                },
                {
                    "name": "SAMSum",
                    "source": "samsum",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://huggingface.co/datasets/samsum",
                    "task_formatter": self._format_samsum_task
                },
                {
                    "name": "XSum",
                    "source": "xsum",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://huggingface.co/datasets/xsum",
                    "task_formatter": self._format_xsum_task
                }
            ],
            "Technical Documentation": [
                {
                    "name": "Stack Exchange",
                    "source": "json:https://archive.org/download/stackexchange/ai.stackexchange.com.7z", 
                    "load_method": "mock",  # Use mock data due to large file size
                    "url": "https://archive.org/details/stackexchange",
                    "task_formatter": self._format_stack_exchange_task
                },
                {
                    "name": "GitHub READMEs",
                    "source": "readmees",  # Mock data
                    "load_method": "mock",
                    "url": "https://github.com/readmejs/readme-dataset",
                    "task_formatter": self._format_readme_task
                },
                {
                    "name": "MDN Web Docs",
                    "source": "mdn",  # Mock data
                    "load_method": "mock",
                    "url": "https://github.com/mdn/content",
                    "task_formatter": self._format_mdn_task
                }
            ],
            "Conversational Response": [
                {
                    "name": "MultiWOZ",
                    "source": "multi_woz_v22",  # HuggingFace dataset name
                    "load_method": "huggingface",
                    "url": "https://github.com/budzianowski/multiwoz",
                    "task_formatter": self._format_multiwoz_task
                },
                {
                    "name": "ConvAI2",
                    "source": "conv_ai_2",  # Mock data
                    "load_method": "mock",
                    "url": "https://github.com/DeepPavlov/convai",
                    "task_formatter": self._format_convai_task
                },
                {
                    "name": "Schema-Guided Dialogue",
                    "source": "schema_guided_dstc8",  # Mock data
                    "load_method": "mock",
                    "url": "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
                    "task_formatter": self._format_schema_guided_task
                }
            ]
        }
    
    def load_datasets(self, categories: Optional[List[str]] = None, 
                     examples_per_category: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """Load examples from real datasets
        
        Args:
            categories: List of categories to load, or None for all
            examples_per_category: Number of examples to load per category
            
        Returns:
            Dictionary of task examples by category
        """
        if categories is None:
            categories = list(self.datasets.keys())
        
        all_examples = {}
        
        for category in categories:
            if category not in self.datasets:
                logger.warning(f"Category {category} not found in datasets")
                continue
            
            category_examples = []
            datasets_in_category = self.datasets[category]
            examples_per_dataset = examples_per_category // len(datasets_in_category)
            
            for dataset_info in datasets_in_category:
                try:
                    logger.info(f"Loading {examples_per_dataset} examples from {dataset_info['name']}")
                    dataset_examples = self._load_dataset_examples(
                        dataset_info, examples_per_dataset
                    )
                    category_examples.extend(dataset_examples)
                except Exception as e:
                    logger.error(f"Error loading dataset {dataset_info['name']}: {e}")
                    # Create mock examples as fallback
                    mock_examples = self._create_mock_examples(
                        dataset_info['name'], category, examples_per_dataset
                    )
                    category_examples.extend(mock_examples)
            
            all_examples[category] = category_examples
        
        return all_examples
    
    def _load_dataset_examples(self, dataset_info: Dict[str, Any], n_examples: int) -> List[Dict[str, Any]]:
        """Load examples from a specific dataset
        
        Args:
            dataset_info: Dataset information
            n_examples: Number of examples to load
            
        Returns:
            List of task examples
        """
        load_method = dataset_info['load_method']
        
        if load_method == 'huggingface':
            return self._load_from_huggingface(dataset_info, n_examples)
        elif load_method == 'csv':
            return self._load_from_csv(dataset_info, n_examples)
        elif load_method == 'json':
            return self._load_from_json(dataset_info, n_examples)
        else:
            # Fallback to mock data
            return self._create_mock_examples(
                dataset_info['name'], dataset_info.get('category', 'Unknown'), n_examples
            )
    
    def _load_from_huggingface(self, dataset_info: Dict[str, Any], n_examples: int) -> List[Dict[str, Any]]:
        """Load examples from a HuggingFace dataset
        
        Args:
            dataset_info: Dataset information
            n_examples: Number of examples to load
            
        Returns:
            List of task examples
        """
        # Remove "huggingface:" prefix if present
        source = dataset_info['source']
        if source.startswith('huggingface:'):
            source = source[len('huggingface:'):]
        
        # Load a small subset to avoid downloading the entire dataset
        try:
            # For most datasets, use test split if available
            dataset = load_dataset(source, split="test")
            if len(dataset) > n_examples:
                dataset = dataset.select(range(n_examples))
        except Exception as e:
            logger.warning(f"Error loading test split: {e}, trying validation split")
            try:
                dataset = load_dataset(source, split="validation")
                if len(dataset) > n_examples:
                    dataset = dataset.select(range(n_examples))
            except Exception as e:
                logger.warning(f"Error loading validation split: {e}, trying train split")
                dataset = load_dataset(source, split="train")
                if len(dataset) > n_examples:
                    dataset = dataset.select(range(n_examples))
        
        # Format the examples using the dataset-specific formatter
        examples = []
        for i, item in enumerate(dataset):
            example = dataset_info['task_formatter'](item, i)
            if example:  # Skip None results
                examples.append(example)
            
            if len(examples) >= n_examples:
                break
        
        return examples
    
    def _load_from_csv(self, dataset_info: Dict[str, Any], n_examples: int) -> List[Dict[str, Any]]:
        """Load examples from a CSV dataset
        
        Args:
            dataset_info: Dataset information
            n_examples: Number of examples to load
            
        Returns:
            List of task examples
        """
        source = dataset_info['source']
        if source.startswith('csv:'):
            url = source[len('csv:'):]
            
            # Download the CSV file
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Load as DataFrame
                csv_data = pd.read_csv(io.StringIO(response.text))
                
                # Save to local file for reference
                dataset_name = dataset_info['name'].lower().replace(' ', '_')
                output_path = os.path.join(self.datasets_dir, f"{dataset_name}.csv")
                csv_data.to_csv(output_path, index=False)
                
                # Format examples
                examples = []
                for i in range(min(n_examples, len(csv_data))):
                    example = dataset_info['task_formatter'](csv_data, i)
                    if example:
                        examples.append(example)
                
                return examples
            except Exception as e:
                logger.error(f"Error downloading CSV: {e}")
                raise
        
        raise ValueError(f"Invalid CSV source: {source}")
    
    def _load_from_json(self, dataset_info: Dict[str, Any], n_examples: int) -> List[Dict[str, Any]]:
        """Load examples from a JSON dataset
        
        Args:
            dataset_info: Dataset information
            n_examples: Number of examples to load
            
        Returns:
            List of task examples
        """
        source = dataset_info['source']
        if source.startswith('json:'):
            url = source[len('json:'):]
            
            # Download the JSON file
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse JSON
                json_data = json.loads(response.text)
                
                # Save to local file for reference
                dataset_name = dataset_info['name'].lower().replace(' ', '_')
                output_path = os.path.join(self.datasets_dir, f"{dataset_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(json_data, f)
                
                # Format examples
                examples = []
                items = json_data
                if isinstance(json_data, dict):
                    items = list(json_data.values())
                
                for i in range(min(n_examples, len(items))):
                    example = dataset_info['task_formatter'](items, i)
                    if example:
                        examples.append(example)
                
                return examples
            except Exception as e:
                logger.error(f"Error downloading JSON: {e}")
                raise
        
        raise ValueError(f"Invalid JSON source: {source}")
    
    def _create_mock_examples(self, dataset_name: str, category: str, n_examples: int) -> List[Dict[str, Any]]:
        """Create mock examples for a dataset
        
        Args:
            dataset_name: Name of the dataset
            category: Category of the dataset
            n_examples: Number of examples to create
            
        Returns:
            List of mock task examples
        """
        logger.warning(f"Creating mock examples for {dataset_name}")
        
        mock_formatters = {
            "Code Generation": self._mock_code_generation_examples,
            "Data Analysis": self._mock_data_analysis_examples,
            "Text Summarization": self._mock_text_summarization_examples,
            "Technical Documentation": self._mock_technical_documentation_examples,
            "Conversational Response": self._mock_conversational_response_examples
        }
        
        formatter = mock_formatters.get(category, self._mock_generic_examples)
        return formatter(dataset_name, n_examples)
    
    # Task formatters for each dataset
    def _format_code_generation_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a code generation task from HumanEval"""
        prompt = item.get("prompt", "")
        task_id = item.get("task_id", f"humaneval-{index}")
        
        # Extract function signature and docstring
        function_parts = prompt.split("def ")
        if len(function_parts) > 1:
            function_def = "def " + function_parts[1].strip()
            docstring_match = re.search(r'"""(.*?)"""', function_def, re.DOTALL)
            description = docstring_match.group(1).strip() if docstring_match else "No description available"
        else:
            function_def = prompt
            description = "Implement this function."
        
        return {
            "task_id": task_id,
            "task_text": f"Implement the following Python function. Make sure to handle edge cases.\n\n{function_def}",
            "task_category": "Code Generation",
            "difficulty": "medium",
            "source": "HumanEval"
        }
    
    def _format_mbpp_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a code generation task from MBPP"""
        problem = item.get("text", item.get("prompt", ""))
        task_id = f"mbpp-{index}"
        
        if not problem:
            return None
        
        return {
            "task_id": task_id,
            "task_text": f"Write a Python function that {problem}. Include test cases.",
            "task_category": "Code Generation",
            "difficulty": "medium",
            "source": "MBPP"
        }
    
    def _format_code_contests_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a code generation task from CodeContests"""
        problem = item.get("problem", "")
        task_id = f"codecontests-{index}"
        
        if not problem:
            return None
        
        # Trim if too long
        if len(problem) > 1000:
            problem = problem[:1000] + "..."
        
        return {
            "task_id": task_id,
            "task_text": f"Solve the following programming problem:\n\n{problem}",
            "task_category": "Code Generation", 
            "difficulty": "hard",
            "source": "CodeContests"
        }
    
    def _format_titanic_task(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Format a data analysis task from Titanic"""
        task_options = [
            "Build a model to predict passenger survival on the Titanic based on age, gender, and passenger class.",
            "Analyze the relationship between passenger class and survival rate in the Titanic dataset.",
            "Create visualizations showing the distribution of age, gender, and survival status in the Titanic dataset.",
            "Perform exploratory data analysis on the Titanic dataset and identify factors correlated with survival.",
            "Calculate the survival rates for different demographic groups in the Titanic dataset."
        ]
        
        task_text = task_options[index % len(task_options)]
        
        # Include a sample of the data in the task
        sample_data = data.head(5).to_string()
        
        return {
            "task_id": f"titanic-{index}",
            "task_text": f"{task_text}\n\nHere's a sample of the data:\n{sample_data}",
            "task_category": "Data Analysis",
            "difficulty": "medium",
            "source": "Titanic"
        }
    
    def _format_housing_task(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Format a data analysis task from Housing"""
        task_options = [
            "Build a regression model to predict house prices using the Boston Housing dataset.",
            "Identify the most important features affecting house prices in the Boston Housing dataset.",
            "Visualize the relationship between crime rate and housing prices in different neighborhoods.",
            "Compare the performance of different regression algorithms on the Boston Housing dataset.",
            "Perform feature engineering to improve prediction accuracy for the Boston Housing dataset."
        ]
        
        task_text = task_options[index % len(task_options)]
        
        # Include a sample of the data in the task
        sample_data = data.head(5).to_string()
        
        return {
            "task_id": f"housing-{index}",
            "task_text": f"{task_text}\n\nHere's a sample of the data:\n{sample_data}",
            "task_category": "Data Analysis",
            "difficulty": "medium",
            "source": "Housing"
        }
    
    def _format_538_task(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Format a data analysis task from FiveThirtyEight"""
        task_options = [
            "Analyze polling trends over time and visualize the results.",
            "Build a predictive model based on polling data from multiple sources.",
            "Create an interactive dashboard showing polling results by demographic groups.",
            "Compare polling accuracy across different polling organizations.",
            "Identify factors that correlate with polling errors in election predictions."
        ]
        
        task_text = task_options[index % len(task_options)]
        
        # Include a sample of the data in the task
        sample_data = data.head(5).to_string()
        
        return {
            "task_id": f"fivethirtyeight-{index}",
            "task_text": f"{task_text}\n\nHere's a sample of the data:\n{sample_data}",
            "task_category": "Data Analysis",
            "difficulty": "medium",
            "source": "FiveThirtyEight"
        }
    
    def _format_cnn_dailymail_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a text summarization task from CNN/Daily Mail"""
        article = item.get("article", "")
        
        if not article:
            return None
        
        # Trim if too long
        if len(article) > 1000:
            article = article[:1000] + "..."
        
        return {
            "task_id": f"cnn-{index}",
            "task_text": f"Summarize the following news article in 3-4 sentences:\n\n{article}",
            "task_category": "Text Summarization",
            "difficulty": "medium",
            "source": "CNN/Daily Mail"
        }
    
    def _format_samsum_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a text summarization task from SAMSum"""
        dialogue = item.get("dialogue", "")
        
        if not dialogue:
            return None
        
        return {
            "task_id": f"samsum-{index}",
            "task_text": f"Summarize the following conversation in a brief paragraph:\n\n{dialogue}",
            "task_category": "Text Summarization",
            "difficulty": "medium",
            "source": "SAMSum"
        }
    
    def _format_xsum_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a text summarization task from XSum"""
        document = item.get("document", "")
        
        if not document:
            return None
        
        # Trim if too long
        if len(document) > 1000:
            document = document[:1000] + "..."
        
        return {
            "task_id": f"xsum-{index}",
            "task_text": f"Write a one-sentence summary of the following news article:\n\n{document}",
            "task_category": "Text Summarization", 
            "difficulty": "medium",
            "source": "XSum"
        }
    
    def _format_stack_exchange_task(self, items: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Format a technical documentation task from Stack Exchange"""
        if index >= len(items):
            return None
        
        item = items[index]
        title = item.get("Title", "Technical question")
        body = item.get("Body", "Question details")
        
        return {
            "task_id": f"stackexchange-{index}",
            "task_text": f"Write comprehensive documentation addressing this technical question:\n\nTitle: {title}\n\n{body}",
            "task_category": "Technical Documentation",
            "difficulty": "hard",
            "source": "Stack Exchange"
        }
    
    def _format_readme_task(self, items: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Format a technical documentation task from GitHub READMEs"""
        # This is using mock data
        project_types = ["Web Framework", "Data Science Library", "Mobile App", "Game Engine", "API Service"]
        project_type = project_types[index % len(project_types)]
        
        return {
            "task_id": f"readme-{index}",
            "task_text": f"Write a comprehensive README.md file for a {project_type} project. Include installation instructions, usage examples, API documentation, and contribution guidelines.",
            "task_category": "Technical Documentation",
            "difficulty": "medium",
            "source": "GitHub READMEs"
        }
    
    def _format_mdn_task(self, items: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Format a technical documentation task from MDN Web Docs"""
        # This is using mock data
        web_topics = ["CSS Flexbox", "JavaScript Promises", "Web APIs", "HTML Canvas", "Browser Storage"]
        topic = web_topics[index % len(web_topics)]
        
        return {
            "task_id": f"mdn-{index}",
            "task_text": f"Create comprehensive technical documentation for {topic}, including explanation, syntax, examples, browser compatibility, and best practices.",
            "task_category": "Technical Documentation",
            "difficulty": "hard",
            "source": "MDN Web Docs"
        }
    
    def _format_multiwoz_task(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Format a conversational response task from MultiWOZ"""
        dialogue = item.get("dialogue", "")
        
        if not dialogue:
            return None
        
        if isinstance(dialogue, list):
            # Format dialogue as text
            dialogue_text = "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {turn}" for i, turn in enumerate(dialogue)])
        else:
            dialogue_text = dialogue
        
        return {
            "task_id": f"multiwoz-{index}",
            "task_text": f"Given the following conversation, provide an appropriate response as the assistant:\n\n{dialogue_text}\n\nAssistant:",
            "task_category": "Conversational Response",
            "difficulty": "medium",
            "source": "MultiWOZ"
        }
    
    def _format_convai_task(self, items: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Format a conversational response task from ConvAI2"""
        # This is using mock data
        personas = [
            "I love hiking and outdoor activities. I have two dogs. I work as a software engineer.",
            "I'm a college student studying psychology. I enjoy cooking and watching movies in my free time.",
            "I'm retired and enjoy gardening. I have three grandchildren. I used to be a teacher.",
            "I'm a professional athlete. I train six days a week. I enjoy reading science fiction."
        ]
        
        conversations = [
            "User: Hi there, how's your day going?\nAssistant: It's going well, just got back from a hike with my dogs. How about yours?\nUser: Pretty good. What kind of work do you do?",
            "User: Hello! What are you up to today?\nAssistant: Just finished cooking a new recipe I learned. Do you like cooking?\nUser: Sometimes. What are you studying in school?",
            "User: Hi, how are you doing?\nAssistant: I'm doing great! Just finished working in my garden. Do you garden?\nUser: No, I don't have much space for that. Do you have any family?",
            "User: Hey there! How's it going?\nAssistant: Good! Just finished my morning training session. How are you?\nUser: I'm fine. What kind of books do you enjoy?"
        ]
        
        persona = personas[index % len(personas)]
        conversation = conversations[index % len(conversations)]
        
        return {
            "task_id": f"convai-{index}",
            "task_text": f"You have the following persona: {persona}\n\nGiven this conversation, respond as the assistant:\n\n{conversation}\n\nAssistant:",
            "task_category": "Conversational Response",
            "difficulty": "medium",
            "source": "ConvAI2"
        }
    
    def _format_schema_guided_task(self, items: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Format a conversational response task from Schema-Guided Dialogue"""
        # This is using mock data
        schemas = [
            "Domain: Restaurant\nSlots: location, cuisine, price_range, reservation_time, number_of_people",
            "Domain: Hotel\nSlots: location, check_in_date, check_out_date, number_of_rooms, star_rating",
            "Domain: Flight\nSlots: origin, destination, departure_date, return_date, number_of_passengers"
        ]
        
        conversations = [
            "User: I'm looking for a restaurant in downtown.\nAssistant: What type of cuisine are you interested in?\nUser: I'd like Italian food, something not too expensive.",
            "User: I need to book a hotel for next weekend.\nAssistant: Where would you like to stay?\nUser: I'm thinking somewhere in Chicago, preferably a 4-star hotel.",
            "User: I want to book a flight from New York to Los Angeles.\nAssistant: When would you like to depart?\nUser: This Friday, and I'll return on Sunday."
        ]
        
        schema = schemas[index % len(schemas)]
        conversation = conversations[index % len(conversations)]
        
        return {
            "task_id": f"schema-guided-{index}",
            "task_text": f"You are a task-oriented dialogue system following this schema:\n{schema}\n\nGiven this conversation, provide an appropriate response that follows the schema:\n\n{conversation}\n\nAssistant:",
            "task_category": "Conversational Response",
            "difficulty": "hard",
            "source": "Schema-Guided Dialogue"
        }
    
    # Mock example generators
    def _mock_code_generation_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate mock code generation examples"""
        tasks = [
            "Write a function to find the kth smallest element in an unsorted array.",
            "Implement a function to check if a string is a palindrome.",
            "Create a function to merge two sorted linked lists.",
            "Write a function to find the longest common subsequence of two strings.",
            "Implement a binary search algorithm for a sorted array.",
            "Create a function that checks if a binary tree is balanced.",
            "Write a function to compute the nth Fibonacci number using dynamic programming.",
            "Implement a function to detect a cycle in a linked list.",
            "Create a function to find all permutations of a string.",
            "Write a function to implement a queue using two stacks."
        ]
        
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": tasks[i % len(tasks)],
                "task_category": "Code Generation",
                "difficulty": "medium",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]
    
    def _mock_data_analysis_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate mock data analysis examples"""
        tasks = [
            "Analyze the correlation between variables in the dataset and visualize the relationships.",
            "Build a linear regression model to predict the target variable and evaluate its performance.",
            "Perform exploratory data analysis and identify outliers in the dataset.",
            "Create visualizations to show the distribution of each feature in the dataset.",
            "Build a classification model to predict categorical outcomes in the dataset.",
            "Analyze time series trends in the dataset and forecast future values.",
            "Perform feature selection to identify the most important predictors.",
            "Create a clustering model to segment the data into meaningful groups.",
            "Compare the performance of different machine learning algorithms on this dataset.",
            "Create an interactive dashboard to visualize key insights from the data."
        ]
        
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": tasks[i % len(tasks)],
                "task_category": "Data Analysis",
                "difficulty": "medium",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]
    
    def _mock_text_summarization_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate mock text summarization examples"""
        tasks = [
            "Summarize the following news article in 3-4 sentences.",
            "Create a one-sentence headline for this news article.",
            "Write a brief summary of the main points in this document.",
            "Condense this long article into a short paragraph.",
            "Extract the key takeaways from this news report.",
            "Write an executive summary of this lengthy document.",
            "Create a bullet-point summary of the main ideas in this text.",
            "Summarize this technical paper for a non-specialist audience.",
            "Create a concise summary that captures the essence of this article.",
            "Summarize the main arguments presented in this opinion piece."
        ]
        
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": tasks[i % len(tasks)],
                "task_category": "Text Summarization",
                "difficulty": "medium",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]
    
    def _mock_technical_documentation_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate mock technical documentation examples"""
        tasks = [
            "Write comprehensive documentation for a REST API endpoint that handles user authentication.",
            "Create a user guide for a command-line tool that processes data files.",
            "Write a tutorial on how to implement a specific algorithm in Python.",
            "Document the architecture of a microservices-based application.",
            "Write installation and setup instructions for a complex software package.",
            "Create API reference documentation for a JavaScript library.",
            "Write a troubleshooting guide for common issues in a web application.",
            "Document best practices for securing a database application.",
            "Create a developer guide for extending a framework with custom modules.",
            "Write detailed documentation for a configuration file format."
        ]
        
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": tasks[i % len(tasks)],
                "task_category": "Technical Documentation",
                "difficulty": "hard",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]
    
    def _mock_conversational_response_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate mock conversational response examples"""
        conversations = [
            "User: I need help finding a restaurant for dinner tonight.\nAssistant: What kind of cuisine are you in the mood for?\nUser: I'm thinking Italian, somewhere not too expensive.",
            "User: Can you help me book a flight?\nAssistant: I'd be happy to help! Where are you flying from and to?\nUser: I want to fly from New York to Los Angeles next weekend.",
            "User: I'm having trouble with my internet connection.\nAssistant: I'm sorry to hear that. Can you tell me what happens when you try to connect?\nUser: My computer says it's connected, but websites won't load.",
            "User: I'm looking for a good book to read.\nAssistant: What genres do you usually enjoy?\nUser: I like science fiction and fantasy novels.",
            "User: How do I reset my password?\nAssistant: I can help with that. Which account are you trying to reset the password for?\nUser: It's for my email account.",
            "User: Can you recommend a movie to watch tonight?\nAssistant: Sure! What kind of movies do you usually enjoy?\nUser: I'm in the mood for something funny."
        ]
        
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": f"Given the following conversation, provide an appropriate response as the assistant:\n\n{conversations[i % len(conversations)]}\n\nAssistant:",
                "task_category": "Conversational Response",
                "difficulty": "medium",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]
    
    def _mock_generic_examples(self, dataset_name: str, n_examples: int) -> List[Dict[str, Any]]:
        """Generate generic mock examples"""
        return [
            {
                "task_id": f"{dataset_name.lower()}-mock-{i}",
                "task_text": f"Task {i} for {dataset_name}: Implement this generic functionality.",
                "task_category": "Unknown",
                "difficulty": "medium",
                "source": dataset_name
            }
            for i in range(n_examples)
        ]


class PromptDataProcessor:
    """
    Process prompt data and create evaluation datasets with real examples
    """
    def __init__(self, csv_path: str, output_dir: str = "data"):
        """
        Initialize the processor with the path to the prompt CSV file
        
        Args:
            csv_path: Path to the CSV file containing system prompts
            output_dir: Directory to store processed data
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.prompts = None
        self.task_examples = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset_manager = RealDatasetManager(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def load_prompts(self) -> pd.DataFrame:
        """
        Load and clean the prompt data from CSV
        
        Returns:
            DataFrame containing cleaned prompt data
        """
        df = pd.read_csv(self.csv_path)
        
        # Clean the data - remove any header rows or empty rows
        df = df[df['Prompt ID'].str.contains(r'^[A-Z]+-\d+', regex=True, na=False)]
        
        # Extract the prompt text from the "Full Prompt Text" column
        # Remove extra quotes if present
        df['Clean Prompt Text'] = df['Full Prompt Text'].apply(
            lambda x: re.sub(r'^"|"$', '', str(x)) if pd.notna(x) else ""
        )
        
        # Add task categories based on Prompt ID prefixes
        df['Task Category'] = df['Prompt ID'].apply(lambda x: x.split('-')[0] if '-' in x else 'Unknown')
        
        self.prompts = df
        logger.info(f"Loaded {len(df)} prompts from {self.csv_path}")
        
        return df
    
    def generate_task_examples(self, n_examples_per_category: int = 50) -> pd.DataFrame:
        """
        Generate task examples using real datasets
        
        Args:
            n_examples_per_category: Number of examples to generate per task category
            
        Returns:
            DataFrame containing task examples
        """
        # Check if we already have prompts loaded
        if self.prompts is None:
            self.load_prompts()
            
        # Map prompt task categories to dataset categories
        category_mapping = {
            'CG': 'Code Generation',
            'DA': 'Data Analysis',
            'TS': 'Text Summarization',
            'TD': 'Technical Documentation',
            'CR': 'Conversational Response'
        }
        
        # Get unique task categories from prompts
        prompt_categories = self.prompts['Task Category'].unique()
        
        # Map to dataset categories
        dataset_categories = []
        for cat in prompt_categories:
            if cat in category_mapping:
                dataset_categories.append(category_mapping[cat])
        
        # Load examples from real datasets
        logger.info(f"Loading examples from real datasets for categories: {dataset_categories}")
        category_examples = self.dataset_manager.load_datasets(
            categories=dataset_categories, 
            examples_per_category=n_examples_per_category
        )
        
        # Flatten examples into DataFrame
        examples = []
        
        for dataset_category, category_data in category_examples.items():
            # Map back to prompt category
            prompt_category = next(
                (pc for pc, dc in category_mapping.items() if dc == dataset_category),
                None
            )
            
            if not prompt_category:
                continue
                
            for example in category_data:
                example["prompt_category"] = prompt_category
                examples.append(example)
        
        # Convert to DataFrame
        examples_df = pd.DataFrame(examples)
        self.task_examples = examples_df
        
        # Save the examples
        examples_df.to_csv(os.path.join(self.output_dir, "task_examples.csv"), index=False)
        logger.info(f"Generated {len(examples_df)} task examples from real datasets")
        
        return examples_df
    
    def create_evaluation_pairs(self) -> pd.DataFrame:
        """
        Create pairs of prompts and task examples for evaluation
        
        Returns:
            DataFrame containing evaluation pairs
        """
        if self.prompts is None:
            self.load_prompts()
            
        if self.task_examples is None:
            self.generate_task_examples()
            
        # Create all combinations of prompts and tasks
        eval_pairs = []
        
        for _, prompt_row in self.prompts.iterrows():
            prompt_id = prompt_row['Prompt ID']
            prompt_category = prompt_row['Task Category']
            prompt_text = prompt_row['Clean Prompt Text']
            
            # Get task examples matching this prompt's category
            matching_tasks = self.task_examples[self.task_examples['prompt_category'] == prompt_category]
            
            # Also include some tasks from other categories (20% of the total)
            other_tasks = self.task_examples[self.task_examples['prompt_category'] != prompt_category]
            if len(other_tasks) > 0:
                num_other = min(int(len(matching_tasks) * 0.2), len(other_tasks))
                sampled_other_tasks = other_tasks.sample(num_other)
                tasks_to_pair = pd.concat([matching_tasks, sampled_other_tasks])
            else:
                tasks_to_pair = matching_tasks
            
            # Create pairs
            for _, task_row in tasks_to_pair.iterrows():
                task_id = task_row['task_id']
                task_text = task_row['task_text']
                task_category = task_row['prompt_category']
                
                # Calculate relevance: 1.0 for matching categories, 0.3 for non-matching
                relevance = 1.0 if prompt_category == task_category else 0.3
                
                eval_pairs.append({
                    "prompt_id": prompt_id,
                    "task_id": task_id,
                    "prompt_text": prompt_text,
                    "task_text": task_text,
                    "prompt_category": prompt_category,
                    "task_category": task_category,
                    "relevance": relevance,
                    "score": None  # Will be filled after evaluation
                })
        
        eval_pairs_df = pd.DataFrame(eval_pairs)
        
        # Save the evaluation pairs
        eval_pairs_df.to_csv(os.path.join(self.output_dir, "evaluation_pairs.csv"), index=False)
        logger.info(f"Created {len(eval_pairs_df)} evaluation pairs")
        
        return eval_pairs_df
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        return self.embedding_model.encode(texts)
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/test splits for model training
        
        Returns:
            Tuple of train and test DataFrames
        """
        # Create evaluation pairs if not done already
        try:
            eval_pairs_df = pd.read_csv(os.path.join(self.output_dir, "evaluation_pairs.csv"))
        except FileNotFoundError:
            eval_pairs_df = self.create_evaluation_pairs()
            
        # Split data into train/test sets (80/20)
        train_df, test_df = train_test_split(
            eval_pairs_df, test_size=0.2, random_state=42, 
            stratify=eval_pairs_df[['prompt_category', 'task_category']]
        )
        
        # Save train/test splits
        train_df.to_csv(os.path.join(self.output_dir, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test_data.csv"), index=False)
        
        logger.info(f"Created train/test split: {len(train_df)} train samples, {len(test_df)} test samples")
        
        return train_df, test_df


if __name__ == "__main__":
    # Example usage
    processor = PromptDataProcessor("System_Prompt.csv")
    processor.load_prompts()
    processor.generate_task_examples()
    train_df, test_df = processor.prepare_training_data()
    
    print(f"Generated {len(processor.task_examples)} task examples from real datasets")
    print(f"Created {len(train_df) + len(test_df)} evaluation pairs")
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")