
import math
from typing import Dict, List, Union


class RecommendationEvaluator:
    """
    Evaluates recommendation quality using precision, recall, and NDCG.
    """

    @staticmethod
    def precision_at_k(recommended: List[Union[int, str]], 
                       relevant: List[Union[int, str]], 
                       k: int = 10) -> float:
        """
        Calculate precision at k.
        
        Args:
            recommended: List of recommended item IDs.
            relevant: List of relevant item IDs.
            k: Number of top recommendations to consider.
        
        Returns:
            Precision score between 0 and 1.
        """
        if not recommended or not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([item for item in recommended_k if item in relevant_set])
        
        return hits / len(recommended_k)

    @staticmethod
    def recall_at_k(recommended: List[Union[int, str]], 
                    relevant: List[Union[int, str]], 
                    k: int = 10) -> float:
        """
        Calculate recall at k.
        
        Args:
            recommended: List of recommended item IDs.
            relevant: List of relevant item IDs.
            k: Number of top recommendations to consider.
        
        Returns:
            Recall score between 0 and 1.
        """
        if not recommended or not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = len([item for item in recommended_k if item in relevant_set])
        
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: List[Union[int, str]], 
                  relevant: List[Union[int, str]], 
                  k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            recommended: List of recommended item IDs.
            relevant: List of relevant item IDs.
            k: Number of top recommendations to consider.
        
        Returns:
            NDCG score between 0 and 1.
        """
        if not recommended or not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant_set:
                dcg += 1.0 / math.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        ideal_len = min(len(relevant), k)
        for i in range(ideal_len):
            idcg += 1.0 / math.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

    @staticmethod
    def evaluate_all(recommendations: Dict[Union[int, str], List[Union[int, str]]], 
                     ground_truth: Dict[Union[int, str], List[Union[int, str]]], 
                     k: int = 10) -> Dict[str, float]:
        """
        Evaluate all metrics and return average scores.
        
        Args:
            recommendations: Mapping from user ID to list of recommended items.
            ground_truth: Mapping from user ID to list of relevant items.
            k: Number of top recommendations to consider.
        
        Returns:
            Dictionary with average precision, recall, and NDCG.
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
            
            rec = recommendations[user_id]
            rel = ground_truth[user_id]
            
            precisions.append(RecommendationEvaluator.precision_at_k(rec, rel, k))
            recalls.append(RecommendationEvaluator.recall_at_k(rec, rel, k))
            ndcgs.append(RecommendationEvaluator.ndcg_at_k(rec, rel, k))
        
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "ndcg": avg_ndcg
        }
