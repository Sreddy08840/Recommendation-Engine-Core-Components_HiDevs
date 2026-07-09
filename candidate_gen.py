
from typing import Dict, List, Set, Union
from collections import defaultdict


class CandidateGenerator:
    """
    Generates recommendation candidates using collaborative, content-based, 
    popularity, and hybrid methods.
    """

    def __init__(self, 
                 user_history: Dict[Union[int, str], List[Union[int, str]]],
                 item_similarity: Dict[Union[int, str], Dict[Union[int, str], float]],
                 popularity: Dict[Union[int, str], float] = None):
        """
        Initialize the candidate generator.
        
        Args:
            user_history: Mapping from user ID to list of item IDs they've interacted with.
            item_similarity: Mapping from item ID to a dictionary of similar item IDs and scores.
            popularity: Optional mapping from item ID to popularity score (default: None).
        """
        self.user_history = user_history
        self.item_similarity = item_similarity
        self.popularity = popularity or {}

    def collaborative_candidates(self, 
                                 user_id: Union[int, str], 
                                 limit: int = 20) -> List[Union[int, str]]:
        """
        Generate candidates using collaborative filtering.
        
        Args:
            user_id: Target user ID.
            limit: Maximum number of candidates to return.
        
        Returns:
            List of candidate item IDs.
        """
        if user_id not in self.user_history:
            return []
        
        user_items = set(self.user_history[user_id])
        user_scores = defaultdict(float)
        
        # Find other users and calculate overlap
        for other_user, other_items in self.user_history.items():
            if other_user == user_id:
                continue
            
            other_set = set(other_items)
            overlap = len(user_items & other_set)
            if overlap == 0:
                continue
            
            # Score items from other users that current user hasn't seen
            for item in other_items:
                if item not in user_items:
                    user_scores[item] += overlap
        
        # Sort by score and take top N
        sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:limit]]

    def content_based_candidates(self, 
                                 user_id: Union[int, str], 
                                 limit: int = 20) -> List[Union[int, str]]:
        """
        Generate candidates using content-based filtering.
        
        Args:
            user_id: Target user ID.
            limit: Maximum number of candidates to return.
        
        Returns:
            List of candidate item IDs.
        """
        if user_id not in self.user_history:
            return []
        
        user_items = set(self.user_history[user_id])
        candidate_scores = defaultdict(float)
        
        for item in user_items:
            if item in self.item_similarity:
                for similar_item, score in self.item_similarity[item].items():
                    if similar_item not in user_items:
                        candidate_scores[similar_item] += score
        
        # Sort by score and take top N
        sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:limit]]

    def popularity_candidates(self, limit: int = 20) -> List[Union[int, str]]:
        """
        Generate candidates based on global popularity.
        
        Args:
            limit: Maximum number of candidates to return.
        
        Returns:
            List of candidate item IDs.
        """
        if not self.popularity:
            return []
        
        sorted_items = sorted(self.popularity.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:limit]]

    def hybrid_candidates(self, 
                          user_id: Union[int, str], 
                          limit: int = 20) -> List[Union[int, str]]:
        """
        Generate hybrid candidates by merging collaborative, content-based, and popularity candidates.
        
        Args:
            user_id: Target user ID.
            limit: Maximum number of candidates to return.
        
        Returns:
            List of candidate item IDs.
        """
        # Get individual candidate lists
        collab = self.collaborative_candidates(user_id, limit)
        content = self.content_based_candidates(user_id, limit)
        pop = self.popularity_candidates(limit)
        
        # Merge while preserving order and removing duplicates
        seen = set()
        hybrid = []
        
        for item in collab + content + pop:
            if item not in seen:
                seen.add(item)
                hybrid.append(item)
                if len(hybrid) >= limit:
                    break
        
        return hybrid[:limit]
