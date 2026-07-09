
import math
from typing import Dict, List, Set, Union


class SimilarityCalculator:
    """
    A utility class for calculating various similarity metrics between data structures.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            vec1: First vector (list of floats).
            vec2: Second vector (list of floats).
        
        Returns:
            Cosine similarity score between 0 and 1.
        
        Raises:
            ValueError: If vectors have different lengths.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for a, b in zip(vec1, vec2):
            dot_product += a * b
            norm1 += a ** 2
            norm2 += b ** 2
        
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        cosine = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, cosine))
    
    @staticmethod
    def jaccard_similarity(set1: Set[Union[int, str]], set2: Set[Union[int, str]]) -> float:
        """
        Calculate the Jaccard similarity between two sets.
        
        Args:
            set1: First set.
            set2: Second set.
        
        Returns:
            Jaccard similarity score between 0 and 1.
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union
    
    @staticmethod
    def pearson_correlation(ratings1: Dict[Union[int, str], float], 
                           ratings2: Dict[Union[int, str], float]) -> float:
        """
        Calculate the Pearson correlation coefficient between two rating dictionaries.
        
        Args:
            ratings1: First user's ratings (item -> rating).
            ratings2: Second user's ratings (item -> rating).
        
        Returns:
            Pearson correlation coefficient between -1 and 1.
        """
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if not common_items:
            return 0.0
        
        n = len(common_items)
        
        # Calculate sums
        sum1 = sum(ratings1[item] for item in common_items)
        sum2 = sum(ratings2[item] for item in common_items)
        
        # Calculate sums of squares
        sum1_sq = sum(ratings1[item] ** 2 for item in common_items)
        sum2_sq = sum(ratings2[item] ** 2 for item in common_items)
        
        # Calculate sum of products
        sum_prod = sum(ratings1[item] * ratings2[item] for item in common_items)
        
        # Calculate Pearson correlation
        numerator = sum_prod - (sum1 * sum2 / n)
        denominator = math.sqrt((sum1_sq - sum1 ** 2 / n) * (sum2_sq - sum2 ** 2 / n))
        
        if denominator == 0:
            return 0.0
        
        return max(-1.0, min(1.0, numerator / denominator))
