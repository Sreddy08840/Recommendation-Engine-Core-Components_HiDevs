
# Recommendation Engine Core Components

A production-ready Python implementation of the core building blocks of a recommendation system. Built with only standard Python libraries, following PEP8 guidelines.


## Project Structure
```
components of a recommendation engine/
├── similarity.py        # Similarity metric calculations
├── candidate_gen.py     # Candidate generation strategies
├── scorer.py            # Weighted scoring and ranking
├── evaluator.py         # Recommendation quality evaluation
└── test.py              # Comprehensive test suite
```


## Features
### 1. Similarity Metrics (`similarity.py`)
- **Cosine Similarity**: Measures cosine of the angle between two vectors
- **Jaccard Similarity**: Measures intersection over union of two sets
- **Pearson Correlation**: Measures linear correlation between two rating dictionaries

### 2. Candidate Generation (`candidate_gen.py`)
- **Collaborative Filtering**: Leverages user-item interaction history
- **Content-Based Filtering**: Uses item-item similarity
- **Popularity-Based**: Recommends globally popular items
- **Hybrid**: Merges all strategies

### 3. Scoring and Ranking (`scorer.py`)
- Weighted scoring system
- Normalized 0-1 scores
- Score explanation
- Multiple scoring factors (relevance, popularity, etc.)

### 4. Evaluation (`evaluator.py`)
- Precision@k
- Recall@k
- NDCG@k (Normalized Discounted Cumulative Gain)


## Installation & Usage
No external dependencies required! Just Python 3.7+


### Running Tests
```bash
cd "components of a recommendation engine"
python test.py
```


## Example Usage
### Similarity Calculator
```python
from similarity import SimilarityCalculator

# Cosine Similarity
vec1 = [1, 2, 3]
vec2 = [1, 2, 3]
cosine = SimilarityCalculator.cosine_similarity(vec1, vec2)
print(cosine)  # Output: 1.0

# Jaccard Similarity
set1 = {1, 2, 3}
set2 = {2, 3, 4}
jaccard = SimilarityCalculator.jaccard_similarity(set1, set2)
print(jaccard)  # Output: 0.5

# Pearson Correlation
ratings1 = {"a":5, "b":4, "c":3}
ratings2 = {"a":5, "b":4, "c":3}
pearson = SimilarityCalculator.pearson_correlation(ratings1, ratings2)
print(pearson)  # Output: 1.0
```


### Candidate Generator
```python
from candidate_gen import CandidateGenerator

user_history = {
    "user1": ["item1", "item2", "item3"],
    "user2": ["item2", "item3", "item4"],
    "user3": ["item3", "item5", "item6"]
}

item_similarity = {
    "item1": {"item4":0.9, "item5":0.7},
    "item2": {"item4":0.8, "item6":0.6}
}

popularity = {"item4":100, "item5":90, "item6":80, "item1":50}

generator = CandidateGenerator(user_history, item_similarity, popularity)
hybrid_candidates = generator.hybrid_candidates("user1", limit=5)
print(hybrid_candidates)
```


### Scorer & Ranker
```python
from scorer import RecommendationScorer

scorer = RecommendationScorer()

def relevance(user_id, item_id, context):
    return 0.9 if item_id in ["item4", "item5"] else 0.5

def popularity_score(user_id, item_id, context):
    pop = {"item4":0.9, "item5":0.8, "item6":0.7}
    return pop.get(item_id, 0.3)

scorer.add_scorer("relevance", relevance, weight=1.0)
scorer.add_scorer("popularity", popularity_score, weight=0.8)

candidates = ["item4", "item5", "item6", "item7"]
ranked = scorer.rank_candidates("user1", candidates, limit=3)
for item in ranked:
    print(f"{item['item']}: {item['score']:.2f}")
```


### Evaluator
```python
from evaluator import RecommendationEvaluator

recommendations = {
    "user1": ["item1", "item2", "item3"]
}

ground_truth = {
    "user1": ["item2", "item4"]
}

metrics = RecommendationEvaluator.evaluate_all(recommendations, ground_truth, k=3)
print(metrics)  # Output: {'precision': 0.333..., 'recall': 0.5, 'ndcg': ...}
```


## Technologies Used
- Python 3.7+
- Standard libraries only: `math`, `typing`, `collections`, `dataclasses`


## Contributing
Contributions, issues, and feature requests are welcome!


## License
MIT
