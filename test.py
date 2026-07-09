
from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator


def test_similarity():
    print("==========================")
    print("Similarity Tests")
    print("==========================")
    passed = 0
    total = 3

    # Cosine Similarity
    try:
        calc = SimilarityCalculator()
        vec1 = [1, 2, 3]
        vec2 = [1, 2, 3]
        assert abs(calc.cosine_similarity(vec1, vec2) - 1.0) < 0.001
        vec3 = [0, 0, 0]
        vec4 = [1, 2, 3]
        assert calc.cosine_similarity(vec3, vec4) == 0.0
        print("[OK] Cosine Similarity Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Cosine Similarity Failed: {e}")

    # Jaccard Similarity
    try:
        set1 = {1, 2, 3}
        set2 = {2, 3, 4}
        assert abs(SimilarityCalculator.jaccard_similarity(set1, set2) - 0.5) < 0.001
        set3 = set()
        set4 = {1, 2}
        assert SimilarityCalculator.jaccard_similarity(set3, set4) == 0.0
        print("[OK] Jaccard Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Jaccard Failed: {e}")

    # Pearson Correlation
    try:
        ratings1 = {"a": 5, "b": 4, "c": 3}
        ratings2 = {"a": 5, "b": 4, "c": 3}
        assert abs(SimilarityCalculator.pearson_correlation(ratings1, ratings2) - 1.0) < 0.001
        ratings3 = {"a": 1, "b": 2}
        ratings4 = {"c": 3, "d": 4}
        assert SimilarityCalculator.pearson_correlation(ratings3, ratings4) == 0.0
        print("[OK] Pearson Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Pearson Failed: {e}")

    print()
    return passed == total


def test_candidate_generator():
    print("==========================")
    print("Candidate Generator Tests")
    print("==========================")
    passed = 0
    total = 4

    # Sample data
    user_history = {
        "user1": ["item1", "item2", "item3"],
        "user2": ["item2", "item3", "item4"],
        "user3": ["item3", "item5", "item6"]
    }
    item_similarity = {
        "item1": {"item4": 0.9, "item5": 0.7},
        "item2": {"item4": 0.8, "item6": 0.6}
    }
    popularity = {"item4": 100, "item5": 90, "item6": 80, "item1": 50}

    gen = CandidateGenerator(user_history, item_similarity, popularity)

    # Collaborative Candidates
    try:
        collab = gen.collaborative_candidates("user1")
        assert len(collab) > 0
        assert "item4" in collab
        print("[OK] Collaborative Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Collaborative Failed: {e}")

    # Content-based Candidates
    try:
        content = gen.content_based_candidates("user1")
        assert len(content) > 0
        print("[OK] Content Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Content Failed: {e}")

    # Popularity Candidates
    try:
        pop = gen.popularity_candidates()
        assert len(pop) > 0
        print("[OK] Popularity Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Popularity Failed: {e}")

    # Hybrid Candidates
    try:
        hybrid = gen.hybrid_candidates("user1")
        assert len(hybrid) > 0
        print("[OK] Hybrid Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Hybrid Failed: {e}")

    print()
    return passed == total


def test_scorer():
    print("==========================")
    print("Scorer Tests")
    print("==========================")
    passed = 0
    total = 2

    scorer = RecommendationScorer()

    # Add sample scorers
    def relevance_scorer(user_id, item_id, context):
        return 0.9 if item_id in ["item4", "item5"] else 0.5

    def popularity_scorer(user_id, item_id, context):
        pop_scores = {"item4": 0.9, "item5": 0.8, "item6": 0.7}
        return pop_scores.get(item_id, 0.3)

    scorer.add_scorer("relevance", relevance_scorer, 1.0)
    scorer.add_scorer("popularity", popularity_scorer, 0.8)

    # Score Calculation
    try:
        result = scorer.calculate_score("user1", "item4")
        assert 0.0 <= result["score"] <= 1.0
        assert "relevance" in result["explanation"][0]
        print("[OK] Score Calculation Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Score Calculation Failed: {e}")

    # Ranking
    try:
        candidates = ["item4", "item5", "item6", "item7"]
        ranked = scorer.rank_candidates("user1", candidates)
        assert len(ranked) == 4
        assert ranked[0]["score"] >= ranked[1]["score"]
        print("[OK] Ranking Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Ranking Failed: {e}")

    print()
    return passed == total


def test_evaluator():
    print("==========================")
    print("Evaluator Tests")
    print("==========================")
    passed = 0
    total = 3

    eval = RecommendationEvaluator()

    # Precision
    try:
        recommended = ["item1", "item2", "item3"]
        relevant = ["item2", "item4"]
        assert abs(eval.precision_at_k(recommended, relevant, k=3) - 1/3) < 0.001
        print("[OK] Precision Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Precision Failed: {e}")

    # Recall
    try:
        recommended = ["item1", "item2", "item3"]
        relevant = ["item2", "item4"]
        assert abs(eval.recall_at_k(recommended, relevant, k=3) - 0.5) < 0.001
        print("[OK] Recall Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Recall Failed: {e}")

    # NDCG
    try:
        recommended = ["item1", "item2", "item3"]
        relevant = ["item2", "item1"]
        assert eval.ndcg_at_k(recommended, relevant, k=3) > 0.0
        print("[OK] NDCG Passed")
        passed += 1
    except Exception as e:
        print(f"[FAIL] NDCG Failed: {e}")

    print()
    return passed == total


def main():
    all_passed = True

    if not test_similarity():
        all_passed = False
    if not test_candidate_generator():
        all_passed = False
    if not test_scorer():
        all_passed = False
    if not test_evaluator():
        all_passed = False

    print("==========================")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("==========================")


if __name__ == "__main__":
    main()
