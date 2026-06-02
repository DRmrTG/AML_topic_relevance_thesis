
#this file contains helper functions for measuring how stable NMF topics are across runs using jaccard similari

import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from prettytable import PrettyTable


#jaccard similarity between two sets of term indices

#sklearn.metrics.jaccard_score is designed for classification tasks, 
#where you're comparing predicted labels against true labels for a set of samples.


def jaccard_binary(set_a, set_b):
    #jaccard similarity = size of intersection / size of union
    
    #converting to sets so we can use & (intersection) and | (union)
    set_x = set(set_a)
    set_y = set(set_b)

    #count how many items are in both sets
    intersection_size = len(set_x & set_y)

    #if nothing overlaps at all, similarity is 0 — return early
    if intersection_size == 0:
        return 0.0

    #count how many items are in either set
    union_size = len(set_x | set_y)

    #this shouldn't happen if we have non-empty sets but just in case
    if union_size == 0:
        return 0.0

    #divide intersection by union to get the similarity score
    similarity = float(intersection_size) / union_size
    return similarity


#build a matrix of pairwise jaccard similarities between two sets of topic rankings


def build_similarity_matrix(rankings1, rankings2):
    #rankings1 and rankings2 are each a list of r lists
    #each inner list contains the top-N term indices for one topic
    #we want to compare every topic in run 1 against every topic in run 2
    #the result is an r x r matrix 

    #number of topics = length of the rankings list
    number_of_topics = len(rankings1)

    #start with a matrix of zeros, shape r x r
    similarity_matrix = np.zeros((number_of_topics, number_of_topics))

    #fill in every cell by comparing topic i from run 1 to topic j from run 2
    for i in range(number_of_topics):
        for j in range(number_of_topics):
            similarity_matrix[i, j] = jaccard_binary(rankings1[i], rankings2[j])

    return similarity_matrix

#hungarian matching 

def pairwise_ats(rankings1, rankings2):
    #this function computes ATS between two runs
    #step 1: build the similarity matrix
    #step 2: find the optimal topic matching using the hungarian algorithm
    #step 3: average the similarity scores of the matched pairs
    #the hungarian algorithm finds the best one-to-one assignment

    #build r x r jaccard similarity matrix
    similarity_matrix = build_similarity_matrix(rankings1, rankings2)

    #linear_sum_assignment minimises cost by default
    #so we negate the matrix to turn it into a maximisation problem
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

    #zip the row and column indices into matched pairs
    #e.g. [(0, 2), (1, 0), (2, 1)] means topic 0 matched to topic 2, etc.
    #converting to list manually instead of using list(zip(...))
    matched_pairs = []
    for k in range(len(row_indices)):
        one_match = (row_indices[k], col_indices[k])
        matched_pairs.append(one_match)

    #compute the mean jaccard similarity across all matched pairs
    #collecting the similarity scores for each matched pair
    matched_similarities = []
    for match in matched_pairs:
        row = match[0]
        col = match[1]
        this_similarity = similarity_matrix[row, col]
        matched_similarities.append(this_similarity)

    #average the similarity scores — this is the ATS score
    total_similarity = 0.0
    for s in matched_similarities:
        total_similarity = total_similarity + s
    ats_score = total_similarity / len(matched_similarities)

    return float(ats_score), matched_pairs, similarity_matrix


def top_n_indices(h_matrix, top):
    #h_matrix has shape r x vocab_size
    #for each topic (row), we want the indices of the top-N highest values
    #these indices correspond to the most important words for that topic

    rankings = []

    #loop through each topic row
    for k in range(h_matrix.shape[0]):
        #get the weights for this topic
        this_row = h_matrix[k]

        #argsort returns indices that would sort smallest to largest
        #[::-1] reverses to get largest first
        #[:top] takes only the first top-N indices
        sorted_indices = np.argsort(this_row)
        sorted_indices_reversed = sorted_indices[::-1]
        top_indices = sorted_indices_reversed[:top]

        #convert to a plain python list and store
        rankings.append(top_indices.tolist())

    return rankings


def evaluate_h_stability(dtm, r, n_runs=10, top=10, verbose=True):
    #this is the main function for the initialization stability script
    #collects H matrices, then computes pairwise ATS across all run pairs
    #returns a dictionary with all the results

    #importing here so this file also works when called standalone
    #i looked this up — importing inside a function is fine in python
    from nnsvdlrc import nnsvdlrc

    #empty lists to collect results from each run
    all_h_matrices = []
    all_final_errors = []
    all_iteration_counts = []
    all_rankings = []

    if verbose:
        print("Running NNSVD-LRC + NMF: r=" + str(r) + ", n_runs=" + str(n_runs) + ", top=" + str(top))
        print("-" * 50)

    #run the model n_runs times
    for i in range(n_runs):
        #run nnsvdlrc — third and fourth return values are unused init matrices
        w_result, h_result, _unused1, _unused2, error_history = nnsvdlrc(dtm, r)

        all_h_matrices.append(h_result)

        #final error = last value in the error history
        final_error = error_history[-1]
        all_final_errors.append(final_error)

        #iteration count = length of error history
        number_of_iters = len(error_history)
        all_iteration_counts.append(number_of_iters)

        #get top-N term indices for this run
        rankings_this_run = top_n_indices(h_result, top)
        all_rankings.append(rankings_this_run)

        if verbose:
            print("  run " + str(i + 1) + " done — final error: " + str(round(final_error, 6)) + "  iters: " + str(number_of_iters))

    if verbose:
        print("")

    #computing pairwise ATS across all unique pairs of runs
    #building pairs manually
    all_pairs = []
    for first in range(n_runs):
        for second in range(n_runs):
            if second > first:
                all_pairs.append([first, second])

    #collecting one ATS score per pair
    all_ats_scores = []
    for pair in all_pairs:
        run_i = pair[0]
        run_j = pair[1]
        ats_score, unused_matches, unused_matrix = pairwise_ats(all_rankings[run_i], all_rankings[run_j])
        all_ats_scores.append(ats_score)

    #convert to numpy array so we can use mean, std etc.
    all_ats_scores = np.array(all_ats_scores)

    if verbose:
        #print a summary table of ATS statistics
        summary_table = PrettyTable(["statistic", "ATS"])
        summary_table.align["statistic"] = "l"

        mean_ats = float(all_ats_scores.mean())
        median_ats = float(np.median(all_ats_scores))
        std_ats = float(all_ats_scores.std())
        min_ats = float(all_ats_scores.min())
        max_ats = float(all_ats_scores.max())

        summary_table.add_row(["mean",   str(round(mean_ats,   4))])
        summary_table.add_row(["median", str(round(median_ats, 4))])
        summary_table.add_row(["std",    str(round(std_ats,    4))])
        summary_table.add_row(["min",    str(round(min_ats,    4))])
        summary_table.add_row(["max",    str(round(max_ats,    4))])

        print("Pairwise ATS across " + str(len(all_pairs)) + " pairs  (top " + str(top) + " terms, r=" + str(r) + ")")
        print(summary_table)

        
        #for each topic, average its stability score across all run pairs
        #this tells us which topics are consistently stable vs which are wobbly
        print("\nPer-topic mean ATS (averaged across all run pairs):")

        #topic_scores[k][pair_idx] = stability of topic k in pair pair_idx
        #building as a list of lists first, then converting to numpy
        topic_scores_list = []
        for k in range(r):
            #one list of scores per topic
            topic_scores_list.append([])

        #fill in topic scores for each pair
        for pair_idx in range(len(all_pairs)):
            run_i = all_pairs[pair_idx][0]
            run_j = all_pairs[pair_idx][1]

            #build the similarity matrix for this pair
            sim_matrix = build_similarity_matrix(all_rankings[run_i], all_rankings[run_j])

            #find the optimal matching for this pair
            row_indices, col_indices = linear_sum_assignment(-sim_matrix)

            #store the similarity score for each matched topic
            for k in range(len(row_indices)):
                topic_row = row_indices[k]
                topic_col = col_indices[k]
                this_score = sim_matrix[topic_row, topic_col]
                topic_scores_list[topic_row].append(this_score)

        #convert to numpy for easy stats
        topic_scores_array = np.array(topic_scores_list)

        #build the per-topic table
        topic_table = PrettyTable(["topic", "mean ATS", "min ATS", "std ATS"])
        for k in range(r):
            topic_mean = round(float(topic_scores_array[k].mean()), 4)
            topic_min  = round(float(topic_scores_array[k].min()),  4)
            topic_std  = round(float(topic_scores_array[k].std()),  4)
            topic_label = "T" + str(k + 1)
            topic_table.add_row([topic_label, str(topic_mean), str(topic_min), str(topic_std)])

        print(topic_table)

    #building the results dictionary manually
    #setting keys one by one
    results = {}
    results["all_scores"]   = all_ats_scores
    results["mean"]         = float(all_ats_scores.mean())
    results["median"]       = float(np.median(all_ats_scores))
    results["std"]          = float(all_ats_scores.std())
    results["min"]          = float(all_ats_scores.min())
    results["max"]          = float(all_ats_scores.max())
    results["Hs"]           = all_h_matrices
    results["errors"]       = all_final_errors
    results["iters"]        = all_iteration_counts
    results["all_rankings"] = all_rankings

    return results

#running file directly if need be


if __name__ == "__main__":
    import sys
    import os
    from scipy.sparse import load_npz

    #check that the user passed a path to the dtm file
    if len(sys.argv) < 2:
        print("Usage: python eval_h_stability.py path/to/dtm.npz [r] [n_runs] [top]")
        sys.exit(1)

    #first argument is the path to the dtm file
    dtm_file_path = sys.argv[1]

    #remaining arguments are optional — use defaults if not provided
    #int(sys.argv[2]) converts the string argument to an integer
    if len(sys.argv) > 2:
        r = int(sys.argv[2])
    else:
        r = 7

    if len(sys.argv) > 3:
        n_runs = int(sys.argv[3])
    else:
        n_runs = 10

    if len(sys.argv) > 4:
        top = int(sys.argv[4])
    else:
        top = 10

    #load the matrix and run the evaluation
    dtm = load_npz(dtm_file_path).astype(float)
    evaluate_h_stability(dtm, r=r, n_runs=n_runs, top=top)