"""Dataset loading utilities for multiple-choice question datasets."""

import os
from typing import List, Dict, Any, Optional, Union
import random
from datasets import load_dataset

class DatasetLoader:
    """Loader for multiple-choice question datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the dataset loader.
        
        Args:
            cache_dir: Optional directory for caching datasets
        """
        self.cache_dir = cache_dir
        self.datasets = {}
    
    def load_dataset(self, 
                    dataset_name: str, 
                    subset: Optional[str] = None, 
                    split: str = "train",
                    limit: Optional[int] = None,
                    shuffle: bool = True,
                    seed: int = 42) -> List[Dict[str, Any]]:
        """Load a dataset from Hugging Face.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            subset: Optional subset of the dataset
            split: Dataset split to load
            limit: Optional limit on number of examples
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            
        Returns:
            List of examples from the dataset
        """
        key = f"{dataset_name}_{subset or ''}_{split}"
        
        try:
            if key in self.datasets:
                print(f"Using cached dataset {key}")
                data = self.datasets[key]
            else:
                print(f"Loading dataset {key}")
                if subset:
                    dataset = load_dataset(dataset_name, subset, split=split, cache_dir=self.cache_dir)
                else:
                    dataset = load_dataset(dataset_name, split=split, cache_dir=self.cache_dir)
                
                # Process the dataset based on its format
                data = self._process_dataset(dataset, dataset_name)
                
                # Store in our datasets dictionary
                self.datasets[key] = data
            
            # Apply limit if needed
            if limit and limit < len(data):
                if shuffle:
                    random.seed(seed)
                    data_sample = random.sample(data, limit)
                else:
                    data_sample = data[:limit]
            else:
                data_sample = data
            
            return data_sample
        
        except Exception as e:
            print(f"Error loading dataset {dataset_name}/{subset}: {e}")
            return []
    
    def _process_dataset(self, dataset, dataset_name: str) -> List[Dict[str, Any]]:
        """Process the dataset into a standardized format.
        
        Args:
            dataset: The HuggingFace dataset
            dataset_name: Name of the dataset for format-specific processing
            
        Returns:
            Processed dataset as a list of dictionaries
        """
        processed_data = []
        
        # Process based on dataset format
        if "mmlu" in dataset_name:
            for item in dataset:
                processed_item = {
                    "question": item["question"],
                    "choices": item["choices"],
                    "answer_index": item["answer"],
                    "dataset": dataset_name,
                    "id": item.get("id", str(len(processed_data)))
                }
                processed_data.append(processed_item)
        
        elif "ai2_arc" in dataset_name:
            for item in dataset:
                choices = item["choices"]["text"]
                answer_index = item["choices"]["label"].index(item["answerKey"])
                
                processed_item = {
                    "question": item["question"],
                    "choices": choices,
                    "answer_index": answer_index,
                    "dataset": dataset_name,
                    "id": item.get("id", str(len(processed_data)))
                }
                processed_data.append(processed_item)
        
        elif "openbookqa" in dataset_name:
            for item in dataset:
                choices = [
                    item["question"]["choices"][i]["text"] 
                    for i in range(len(item["question"]["choices"]))
                ]
                answer_key = item["answerKey"]
                answer_index = ord(answer_key) - ord("A")
                
                processed_item = {
                    "question": item["question"]["stem"],
                    "choices": choices,
                    "answer_index": answer_index,
                    "dataset": dataset_name,
                    "id": item.get("id", str(len(processed_data)))
                }
                processed_data.append(processed_item)
        
        else:
            # Generic processing for unknown datasets
            for i, item in enumerate(dataset):
                # Try to extract the necessary fields
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer_index = item.get("answer", 0)
                
                processed_item = {
                    "question": question,
                    "choices": choices,
                    "answer_index": answer_index,
                    "dataset": dataset_name,
                    "id": item.get("id", str(i))
                }
                processed_data.append(processed_item)
        
        return processed_data