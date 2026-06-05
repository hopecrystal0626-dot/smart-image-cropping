import numpy as np
from typing import List, Tuple

from composition.thirds import compute_thirds_score_batch
from composition.balance import compute_balance_score_batch
from composition.whitespace import compute_whitespace_score_batch

Bbox = Tuple[int, int, int, int]


class CompositionScorer:

    def __init__(self, weights=(0.4, 0.2, 0.4)):
        self.w_thirds = weights[0]
        self.w_balance = weights[1]
        self.w_whitespace = weights[2]

    def compute_scores(self, image, candidates: List[Bbox]):

        if not candidates:
            return []

        thirds = compute_thirds_score_batch(image, candidates)
        balance = compute_balance_score_batch(image, candidates)
        whitespace = compute_whitespace_score_batch(image, candidates)

        scores = []
        for t, b, w in zip(thirds, balance, whitespace):
            score = (
                self.w_thirds * t +
                self.w_balance * b +
                self.w_whitespace * w
            )
            scores.append(float(score))

        return scores