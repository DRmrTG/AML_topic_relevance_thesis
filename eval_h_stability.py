#!/usr/bin/env python
"""
Evaluate the stability of the topic-term matrix H across repeated runs of
NNSVD-LRC initialisation + NMF, using Average Term Stability (ATS).

Designed to be called from a notebook cell alongside:
    from nnsvdlrc import nnsvdlrc

Usage (from notebook):
    from eval_h_stability import evaluate_h_stability
    results = evaluate_h_stability(dtm, r=7, n_runs=10, top=10)
"""

import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from prettytable import PrettyTable


# --------------------------------------------------------------
# Jaccard similarity between two term sets
# --------------------------------------------------------------

def jaccard_binary(set_a, set_b):
    """Binary Jaccard similarity between two sets of term indices."""
    sx = set(set_a)
    sy = set(set_b)
    numer = len(sx & sy)
    if numer == 0:
        return 0.0
    denom = len(sx | sy)
    if denom == 0:
        return 0.0
    return float(numer) / denom


# --------------------------------------------------------------
# Build pairwise Jaccard similarity matrix between two ranking sets
# --------------------------------------------------------------

def build_similarity_matrix(rankings1, rankings2):
    """
    Construct an r x r matrix of pairwise Jaccard similarities between
    the topic-term sets of two runs.

    Parameters
    ----------
    rankings1, rankings2 : list of list of int
        Each is a list of r topic-term index lists (top-N term indices per topic).

    Returns
    -------
    S : np.ndarray, shape (r, r)
    """
    r = len(rankings1)
    S = np.zeros((r, r))
    for i in range(r):
        for j in range(r):
            S[i, j] = jaccard_binary(rankings1[i], rankings2[j])
    return S


# --------------------------------------------------------------
# Hungarian matching and ATS score for a pair of runs
# --------------------------------------------------------------

def pairwise_ats(rankings1, rankings2):
    """
    Compute the Average Term Stability between two ranking sets via
    optimal Hungarian alignment of topics, then mean Jaccard over matched pairs.

    Returns
    -------
    score : float
        Mean Jaccard similarity across optimally matched topic pairs.
    matches : list of (int, int)
        The matched topic index pairs (run1_topic, run2_topic).
    S : np.ndarray
        The full similarity matrix, for per-topic inspection.
    """
    S = build_similarity_matrix(rankings1, rankings2)
    # scipy's linear_sum_assignment minimises cost, so negate for maximisation
    row_ind, col_ind = linear_sum_assignment(-S)
    matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    score = float(np.mean([S[r, c] for r, c in matches]))
    return score, matches, S


# --------------------------------------------------------------
# Extract top-N term indices from a row of H
# --------------------------------------------------------------

def top_n_indices(H, top):
    """
    For each row of H (shape r x vocab), return the indices of the top-N
    highest-loading terms.

    Returns
    -------
    rankings : list of list of int, length r
    """
    rankings = []
    for k in range(H.shape[0]):
        indices = np.argsort(H[k])[::-1][:top]
        rankings.append(indices.tolist())
    return rankings


# --------------------------------------------------------------
# Main evaluation function
# --------------------------------------------------------------

def evaluate_h_stability(dtm, r, n_runs=10, top=10, verbose=True):
    """
    Run NNSVD-LRC + NMF n_runs times on dtm, extract top-N term indices
    from H after each run, then compute pairwise ATS across all run pairs.

    Parameters
    ----------
    dtm : scipy sparse matrix, shape (vocab, docs)
        Document-term matrix.
    r : int
        Number of topics (factorisation rank).
    n_runs : int
        Number of independent runs.
    top : int
        Number of top terms per topic to use for ATS.
    verbose : bool
        Print progress and results.

    Returns
    -------
    dict with keys:
        'all_scores'   : np.ndarray of pairwise ATS scores
        'mean'         : float
        'median'       : float
        'std'          : float
        'min'          : float
        'max'          : float
        'Hs'           : list of H matrices from each run
        'errors'       : list of final KL errors
        'iters'        : list of iteration counts
        'all_rankings' : list of top-N index sets per run
    """
    # Import here so the script is usable standalone or from a notebook
    from nnsvdlrc import nnsvdlrc

    Hs, errors, iters, all_rankings = [], [], [], []

    if verbose:
        print(f"Running NNSVD-LRC + NMF: r={r}, n_runs={n_runs}, top={top}")
        print("-" * 50)

    for i in range(n_runs):
        W, H, _, _, e = nnsvdlrc(dtm, r)
        Hs.append(H)
        errors.append(e[-1])
        iters.append(len(e))
        rankings = top_n_indices(H, top)
        all_rankings.append(rankings)
        if verbose:
            print(f"  Run {i+1:>2}/{n_runs}  final_error={e[-1]:.6f}  iters={len(e)}")

    if verbose:
        print()

    # Pairwise ATS across all (n_runs choose 2) pairs
    run_pairs = list(combinations(range(n_runs), 2))
    all_scores = []

    for (i, j) in run_pairs:
        score, _, _ = pairwise_ats(all_rankings[i], all_rankings[j])
        all_scores.append(score)

    all_scores = np.array(all_scores)

    if verbose:
        tab = PrettyTable(["statistic", "ATS"])
        tab.align["statistic"] = "l"
        tab.add_row(["mean",   f"{all_scores.mean():.4f}"])
        tab.add_row(["median", f"{np.median(all_scores):.4f}"])
        tab.add_row(["std",    f"{all_scores.std():.4f}"])
        tab.add_row(["min",    f"{all_scores.min():.4f}"])
        tab.add_row(["max",    f"{all_scores.max():.4f}"])
        print(f"Pairwise ATS across {len(run_pairs)} pairs  (top {top} terms, r={r})")
        print(tab)

        # Per-topic breakdown: average stability of each topic across all pairs
        print("\nPer-topic mean ATS (averaged across all run pairs):")
        topic_scores = np.zeros((r, len(run_pairs)))
        for pair_idx, (i, j) in enumerate(run_pairs):
            S = build_similarity_matrix(all_rankings[i], all_rankings[j])
            row_ind, col_ind = linear_sum_assignment(-S)
            for k, (ri, ci) in enumerate(zip(row_ind, col_ind)):
                topic_scores[ri, pair_idx] = S[ri, ci]

        topic_tab = PrettyTable(["topic", "mean ATS", "min ATS", "std ATS"])
        for k in range(r):
            topic_tab.add_row([
                f"T{k+1}",
                f"{topic_scores[k].mean():.4f}",
                f"{topic_scores[k].min():.4f}",
                f"{topic_scores[k].std():.4f}",
            ])
        print(topic_tab)

    return {
        "all_scores":   all_scores,
        "mean":         float(all_scores.mean()),
        "median":       float(np.median(all_scores)),
        "std":          float(all_scores.std()),
        "min":          float(all_scores.min()),
        "max":          float(all_scores.max()),
        "Hs":           Hs,
        "errors":       errors,
        "iters":        iters,
        "all_rankings": all_rankings,
    }


# --------------------------------------------------------------
# Standalone entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from scipy.sparse import load_npz

    if len(sys.argv) < 2:
        print("Usage: python eval_h_stability.py path/to/dtm.npz [r] [n_runs] [top]")
        sys.exit(1)

    dtm_path = sys.argv[1]
    r      = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    top    = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    dtm = load_npz(dtm_path).astype(float)
    evaluate_h_stability(dtm, r=r, n_runs=n_runs, top=top)