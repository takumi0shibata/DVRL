"""General utility functions for the project."""

import numpy as np
import random
import os
import torch

def set_seed(seed):
    # fix random seed
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_overall_score_range():
    return {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}

def get_overall_score_range_for_rubric():
    return {
    1: (1, 6),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 30),
    8: (0, 60)
}


def get_analytic_score_range():
    return {
    1: (1, 6),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 6),
    8: (2, 12)
}


def get_min_max_score_vector():
    return {1:
                {'max': [12, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1],
                 'min': [2, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]},
            2:
                {'max': [6, 6, 6, 6, 6, 6, -1, -1, -1, -1, -1],
                 'min': [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]},
            3:
                {'max': [3, 3, -1, -1, -1, -1, 3, 3, 3, -1, -1],
                 'min': [0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1]},
            4:
                {'max': [3, 3, -1, -1, -1, -1, 3, 3, 3, -1, -1],
                 'min': [0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1]},
            5:
                {'max': [4, 4, -1, -1, -1, -1, 4, 4, 4, -1, -1],
                 'min': [0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1]},
            6:
                {'max': [4, 4, -1, -1, -1, -1, 4, 4, 4, -1, -1],
                 'min': [0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1]},
            7:
                {'max': [30, 6, 6, -1, -1, 6, -1, -1, -1, 6, -1],
                 'min': [0, 0, 0, -1, -1, 0, -1, -1, -1, 0, -1]},
            8:
                {'max': [60, 12, 12, 12, 12, 12, -1, -1, -1, -1, 12],
                 'min': [0, 2, 2, 2, 2, 2, -1, -1, -1, -1, 2]}}


def get_attribute_mask_vector(prompt_id):
    scores = get_min_max_score_vector()[prompt_id]['max']
    scores = np.array(scores)
    return scores != -1


def compute_mask(overall_range, analytic_range, num_item):
    mask_overall = np.zeros((1, overall_range))
    mask_analytic = np.zeros((num_item-1, analytic_range))
    if mask_overall.shape[1] == mask_analytic.shape[1]:
        inf_mask = np.concatenate([mask_overall, mask_analytic], axis=0)
    else:
        inf_arr = np.full((num_item-1, overall_range-analytic_range), -1e9)
        mask_analytic = np.concatenate([mask_analytic, inf_arr], axis=1)
        inf_mask = np.concatenate([mask_overall, mask_analytic], axis=0)

    beta_mask = inf_mask.copy()
    beta_mask[beta_mask==0] = 1.0
    beta_mask[beta_mask==-1e9] = 0.0

    return inf_mask, beta_mask


def get_score_vector_positions():
    return {
        'score': 0,
        'content': 1,
        'organization': 2,
        'word_choice': 3,
        'sentence_fluency': 4,
        'conventions': 5,
        'prompt_adherence': 6,
        'language': 7,
        'narrativity': 8,
        'style': 9,
        'voice': 10
    }


def get_min_max_scores():
    return {
        1: {'score': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6),
        'style': (0, 6)
        },
        8: {'score': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12),
            'voice': (2, 12)
            }}

def get_min_max_scores_for_rubric():
    return {
        1: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        2: {'score': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
            'sentence_fluency': (1, 6), 'conventions': (1, 6)},
        3: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        4: {'score': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
        5: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        6: {'score': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
        7: {'score': (0, 30), 'content': (0, 3), 'organization': (0, 6), 'conventions': (0, 6),
        'style': (0, 6)
        },
        8: {'score': (0, 60), 'content': (1, 6), 'organization': (2, 12), 'word_choice': (2, 12),
            'sentence_fluency': (2, 12), 'conventions': (2, 12),
            'voice': (2, 12)
            }}



def get_scaled_down_scores(scores, prompts):
    score_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    for score_vector, prompt in score_prompts:
        rescaled_score_vector = [-1] * len(score_positions)
        for ind, att_val in enumerate(score_vector):
            if att_val != -1:
                attribute_name = list(score_positions.keys())[list(score_positions.values()).index(ind)]
                min_val = min_max_scores[prompt][attribute_name][0]
                max_val = min_max_scores[prompt][attribute_name][1]
                scaled_score = (att_val - min_val) / (max_val - min_val)
                rescaled_score_vector[ind] = scaled_score
        scaled_score_list.append(rescaled_score_vector)
    assert len(scaled_score_list) == len(scores)
    for scores in scaled_score_list:
        assert min(scores) >= -1
        assert max(scores) <= 1
    return scaled_score_list


def get_single_scaled_down_score(scores, prompts, attribute_name, rubric=False):
    if rubric:
        min_max_scores = get_min_max_scores_for_rubric()
    else:
        min_max_scores = get_min_max_scores()
    score_prompts = zip(scores, prompts)
    scaled_score_list = []
    for score_vector, prompt in score_prompts:
        for ind, att_val in enumerate(score_vector):
            min_val = min_max_scores[prompt][attribute_name][0]
            max_val = min_max_scores[prompt][attribute_name][1]
            scaled_score = (att_val - min_val) / (max_val - min_val)
        scaled_score_list.append([scaled_score])
    assert len(scaled_score_list) == len(scores)
    return scaled_score_list


def rescale_tointscore(scaled_scores, set_ids):
    '''
    rescale scaled scores range[0,1] to original integer scores based on  their set_ids
    :param scaled_scores: list of scaled scores range [0,1] of essays
    :param set_ids: list of corresponding set IDs of essays, integer from 1 to 8
    '''
    if isinstance(set_ids, int):
        prompt_id = set_ids
        set_ids = np.ones(scaled_scores.shape[0],) * prompt_id
    assert scaled_scores.shape[0] == len(set_ids)
    int_scores = np.zeros((scaled_scores.shape[0], 1))
    for k, i in enumerate(set_ids):
        assert i in range(1, 9)
        # TODO
        if i == 1:
            minscore = 2
            maxscore = 12
        elif i == 2:
            minscore = 1
            maxscore = 6
        elif i in [3, 4]:
            minscore = 0
            maxscore = 3
        elif i in [5, 6]:
            minscore = 0
            maxscore = 4
        elif i == 7:
            minscore = 0
            maxscore = 30
        elif i == 8:
            minscore = 0
            maxscore = 60
        else:
            print ("Set ID error")

        int_scores[k] = scaled_scores[k]*(maxscore-minscore) + minscore
    return np.around(int_scores).astype(int)


def rescale_single_attribute(scores, set_ids, attribute_name, rubric=False):
    if rubric:
        min_max_scores = get_min_max_scores_for_rubric()
    else:
        min_max_scores = get_min_max_scores()
    score_id_combined = list(zip(scores, set_ids))
    rescaled_scores = []
    for score, set_id in score_id_combined:
        min_score = min_max_scores[set_id][attribute_name][0]
        max_score = min_max_scores[set_id][attribute_name][1]
        rescaled_score = score * (max_score - min_score) + min_score
        rescaled_scores.append(np.around(rescaled_score).astype(int))
    return np.array(rescaled_scores)


def separate_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {att: [] for att in score_vector_positions.keys()}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            att_position = score_vector_positions[relevant_attribute]
            individual_att_scores_dict[relevant_attribute].append(att_scores[att_position])
    return individual_att_scores_dict


def separate_and_rescale_attributes_for_scoring(scores, set_ids):
    score_vector_positions = get_score_vector_positions()
    min_max_scores = get_min_max_scores()
    individual_att_scores_dict = {}
    score_set_comb = list(zip(scores, set_ids))
    for att_scores, set_id in score_set_comb:
        for relevant_attribute in min_max_scores[set_id].keys():
            min_score = min_max_scores[set_id][relevant_attribute][0]
            max_score = min_max_scores[set_id][relevant_attribute][1]
            att_position = score_vector_positions[relevant_attribute]
            att_score = att_scores[att_position]
            rescaled_score = att_score * (max_score - min_score) + min_score
            try:
                individual_att_scores_dict[relevant_attribute].append(np.around(rescaled_score).astype(int))
            except KeyError:
                individual_att_scores_dict[relevant_attribute] = [np.around(rescaled_score).astype(int)]
    return individual_att_scores_dict


def pad_flat_text_sequences(index_sequences, max_essay_len):
    X = np.empty([len(index_sequences), max_essay_len], dtype=np.int32)

    for i, essay in enumerate(index_sequences):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)
        for j in range(num):
            word_id = sequence_ids[j]
            X[i, j] = word_id
        length = len(sequence_ids)
        X[i, length:] = 0
    return X


def pad_hierarchical_text_sequences(index_sequences, max_sentnum, max_sentlen):
    X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

    for i in range(len(index_sequences)):
        sequence_ids = index_sequences[i]
        num = len(sequence_ids)

        for j in range(num):
            word_ids = sequence_ids[j]
            length = len(word_ids)
            for k in range(length):
                wid = word_ids[k]
                X[i, j, k] = wid
            X[i, j, length:] = 0

        X[i, num:, :] = 0
    return X

def flatten_hierarchical_sequences(data: list) -> list:
    """
    Flatten hierarchical text sequences to a flat list by removing padding.
    
    Args:
    - data (list): The input data with shape [batch, max_sentence_num, max_sentence_length].
    
    Returns:
    - list: The flattened data with variable lengths.
    """
    flattened_data = []
    for document in data:
        # Filter out the padding values from each sentence and flatten
        flattened_document = [word for sentence in document for word in sentence]
        flattened_data.append(flattened_document)
    
    # Since the lengths are variable, we return a list of lists
    return flattened_data

def pad_text_sequences(sequences, max_length):
    padding_value = 0
    padded_sequences = []

    for seq in sequences:
        padded_seq = seq + [padding_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences)


def get_attribute_masks(score_matrix):
    mask_value = -1
    mask = np.cast['int32'](np.not_equal(score_matrix, mask_value))
    return mask


def load_word_embedding_dict(embedding_path):
    print("Loading GloVe ...")
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim, True


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim])
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    for word in word_alphabet:
        ww = word.lower() if caseless else word
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        embedd_table[word_alphabet[word], :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    print("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table
