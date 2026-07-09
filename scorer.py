
from typing import Dict, List, Union, Callable, Any
from dataclasses import dataclass


@dataclass
class ScoringFunction:
    name: str
    function: Callable[[Any, Any, Dict], float]
    weight: float


class RecommendationScorer:
    """
    Scores and ranks recommendation candidates using multiple weighted factors.
    """

    def __init__(self):
        self.scorers: List[ScoringFunction] = []

    def add_scorer(self, name: str, function: Callable[[Any, Any, Dict], float], weight: float = 1.0):
        """
        Add a new scoring function.
        
        Args:
            name: Name of the scorer (for explanation).
            function: Function that takes (user_id, item_id, context) and returns a score 0-1.
            weight: Weight of this scorer in the final score.
        """
        self.scorers.append(ScoringFunction(name=name, function=function, weight=weight))

    def calculate_score(self, 
                        user_id: Union[int, str], 
                        item_id: Union[int, str], 
                        context: Dict = None) -> Dict[str, Any]:
        """
        Calculate the weighted score for an item.
        
        Args:
            user_id: Target user ID.
            item_id: Item to score.
            context: Optional additional context data.
        
        Returns:
            Dictionary with item, score, and explanation.
        """
        context = context or {}
        total_weight = 0.0
        weighted_sum = 0.0
        explanation = []
        
        for scorer in self.scorers:
            score = scorer.function(user_id, item_id, context)
            score = max(0.0, min(1.0, score))
            weighted_sum += score * scorer.weight
            total_weight += scorer.weight
            explanation.append(f"{scorer.name}:{score:.2f}")
        
        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = weighted_sum / total_weight
        
        final_score = max(0.0, min(1.0, final_score))
        
        return {
            "item": item_id,
            "score": final_score,
            "explanation": explanation
        }

    def rank_candidates(self, 
                        user_id: Union[int, str], 
                        candidates: List[Union[int, str]], 
                        limit: int = 20, 
                        context: Dict = None) -> List[Dict[str, Any]]:
        """
        Score and rank a list of candidates.
        
        Args:
            user_id: Target user ID.
            candidates: List of item IDs to rank.
            limit: Maximum number of items to return.
            context: Optional additional context data.
        
        Returns:
            List of scored items sorted by score descending.
        """
        scored_items = [self.calculate_score(user_id, item, context) for item in candidates]
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        return scored_items[:limit]
