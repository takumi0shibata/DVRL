import numpy as np

from utils.create_embedding_feautres import (
    create_embedding_features,
    load_data,
    normalize_scores,
)
from utils.dvrl_utils import get_dev_sample
from utils.read_data import read_essays_single_score, read_pos_vocab
from utils.general_utils import (
    get_single_scaled_down_score,
    pad_hierarchical_text_sequences
)

def load_data_DVRL(
        data_path: str,
        attribute_name: str,
        embedding_model: str,
        device: str,
        devsize: int | float = 30
    ) -> dict[str, np.ndarray]:

    # Load ASAP data
    train_data, val_data, test_data = create_embedding_features(
        data_path,
        attribute_name,
        embedding_model,
        device
    )
    x_source = np.concatenate([train_data['essay'], val_data['essay']])
    y_source = np.concatenate([train_data['normalized_label'], val_data['normalized_label']])

    # split test data into dev and test
    x_dev, x_target, y_dev, y_target, dev_ids, target_ids = get_dev_sample(
        test_data['essay'],
        test_data['normalized_label'],
        dev_size=devsize,
    )

    # print info
    print('================================')
    print('X_source: ', x_source.shape)
    print('Y_source: ', y_source.shape)
    print('Y_source max: ', np.max(y_source))
    print('Y_source min: ', np.min(y_source))

    print('================================')
    print('X_dev: ', x_dev.shape)
    print('Y_dev: ', y_dev.shape)
    print('Y_dev max: ', np.max(y_dev))
    print('Y_dev min: ', np.min(y_dev))

    print('================================')
    print('X_target: ', x_target.shape)
    print('Y_target: ', y_target.shape)
    print('Y_target max: ', np.max(y_target))
    print('Y_target min: ', np.min(y_target))
    print('================================')

    return {
        'x_source': x_source,
        'y_source': y_source,
        'x_dev': x_dev,
        'y_dev': y_dev,
        'x_target': x_target,
        'y_target': y_target,
        'dev_ids': dev_ids,
        'target_ids': target_ids
    }


def load_data_PAES(
    data_path: str,
    attribute_name: str,
    embedding_model: str,
    device: str,
    features_path: str = 'data/hand_crafted_v3.csv',
    readability_path: str = 'data/allreadability.pickle',
    devsize: int | float = 30
):
    _, _, test_data = create_embedding_features(
        data_path,
        attribute_name,
        embedding_model,
        device
    )
    _, _, _, _, dev_idx, test_idx = get_dev_sample(
        test_data['essay'],
        test_data['normalized_label'],
        dev_size=devsize
    )

    read_configs = {
        'train_path': data_path + 'train.pk',
        'dev_path': data_path + 'dev.pk',
        'test_path': data_path + 'test.pk',
        'features_path': features_path,
        'readability_path': readability_path
    }
    # Read data
    pos_vocab = read_pos_vocab(read_configs)
    train_data, dev_data, test_data = read_essays_single_score(read_configs, pos_vocab, attribute_name)


    # Get max sentence length and max sentence number
    max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])

    # Scale down the scores
    train_data['y_scaled'] = get_single_scaled_down_score(train_data['data_y'], train_data['prompt_ids'], attribute_name)
    dev_data['y_scaled'] = get_single_scaled_down_score(dev_data['data_y'], dev_data['prompt_ids'], attribute_name)
    test_data['y_scaled'] = get_single_scaled_down_score(test_data['data_y'], test_data['prompt_ids'], attribute_name)

    # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
    X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

    X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))

    # convert to tensor
    X_source = np.concatenate([X_train_pos, X_dev_pos], axis=0)
    X_dev = X_test_pos[dev_idx]
    X_target = X_test_pos[test_idx]

    X_source_linguistic_features = np.concatenate([train_data['features_x'], dev_data['features_x']], axis=0)
    X_dev_linguistic_features = np.array(test_data['features_x'])[dev_idx]
    X_target_linguistic_features = np.array(test_data['features_x'])[test_idx]

    X_source_readability = np.concatenate([train_data['readability_x'], dev_data['readability_x']], axis=0)
    X_dev_readability = np.array(test_data['readability_x'])[dev_idx]
    X_target_readability = np.array(test_data['readability_x'])[test_idx]

    X_source_features = np.concatenate([X_source_linguistic_features, X_source_readability], axis=1)
    X_dev_features = np.concatenate([X_dev_linguistic_features, X_dev_readability], axis=1)
    X_target_features = np.concatenate([X_target_linguistic_features, X_target_readability], axis=1)

    Y_source = np.concatenate([train_data['y_scaled'], dev_data['y_scaled']], axis=0)
    Y_dev = np.array(test_data['y_scaled'])[dev_idx]
    Y_target = np.array(test_data['y_scaled'])[test_idx]

    source_essay_set = np.concatenate([train_data['prompt_ids'], dev_data['prompt_ids']], axis=0)
    dev_essay_set = np.array(test_data['prompt_ids'])[dev_idx]
    target_essay_set = np.array(test_data['prompt_ids'])[test_idx]

    X_source_set = (X_source, X_source_features, source_essay_set)
    X_dev_set = (X_dev, X_dev_features, dev_essay_set)
    X_target_set = (X_target, X_target_features, target_essay_set)

    # print info
    print('================================')
    print('X_source: ', X_source.shape)
    print('X_source_linguistic_features: ', X_source_linguistic_features.shape)
    print('X_source_readability: ', X_source_readability.shape)
    print('Y_source: ', Y_source.shape)
    print('Y_source max: ', np.max(Y_source))
    print('Y_source min: ', np.min(Y_source))

    print('================================')
    print('X_dev: ', X_dev.shape)
    print('X_dev_linguistic_features: ', X_dev_linguistic_features.shape)
    print('X_dev_readability: ', X_dev_readability.shape)
    print('Y_dev: ', Y_dev.shape)
    print('Y_dev max: ', np.max(Y_dev))
    print('Y_dev min: ', np.min(Y_dev))

    print('================================')
    print('X_target: ', X_target.shape)
    print('X_target_linguistic_features: ', X_target_linguistic_features.shape)
    print('X_target_readability: ', X_target_readability.shape)
    print('Y_target: ', Y_target.shape)
    print('Y_target max: ', np.max(Y_target))
    print('Y_target min: ', np.min(Y_target))
    print('================================')

    return {
        'pos_vocab': pos_vocab,
        'x_source': X_source_set,
        'x_source_linguistic_features': X_source_linguistic_features,
        'x_source_readability': X_source_readability,
        'y_source': Y_source,
        'x_dev': X_dev_set,
        'x_dev_linguistic_features': X_dev_linguistic_features,
        'x_dev_readability': X_dev_readability,
        'y_dev': Y_dev,
        'x_target': X_target_set,
        'x_target_linguistic_features': X_target_linguistic_features,
        'x_target_readability': X_target_readability,
        'y_target': Y_target,
        'max_sentnum': max_sentnum,
        'max_sentlen': max_sentlen,
    }


def load_data_Transformers(
        data_path: str,
        attribute_name: str,
        embedding_model: str,
        device: str,
        devsize: int | float = 30
):
    data = load_data(data_path)

    # Load ASAP data
    _, _, test_data = create_embedding_features(
        data_path,
        attribute_name,
        embedding_model,
        device
    )
    _, _, _, _, dev_idx, test_idx = get_dev_sample(
        test_data['essay'],
        test_data['normalized_label'],
        dev_size=devsize
    )

    x_source = np.concatenate([data['train']['feature'], data['dev']['feature']])
    source_prompts = np.concatenate([data['train']['essay_set'], data['dev']['essay_set']])
    y_source = np.concatenate([data['train']['label'], data['dev']['label']])
    y_source = normalize_scores(y_source, source_prompts, attribute_name)

    x_dev = np.array(data['test']['feature'])[dev_idx]
    dev_prompts = np.array(data['test']['essay_set'])[dev_idx]
    y_dev = np.array(data['test']['label'])[dev_idx]
    y_dev = normalize_scores(y_dev, dev_prompts, attribute_name)

    x_target = np.array(data['test']['feature'])[test_idx]
    target_prompts =np.array(data['test']['essay_set'])[test_idx]
    y_target = np.array(data['test']['label'])[test_idx]
    y_target = normalize_scores(y_target, target_prompts, attribute_name)

    # print info
    print('================================')
    print('X_source: ', x_source.shape)
    print('Y_source: ', y_source.shape)
    print('Y_source max: ', np.max(y_source))
    print('Y_source min: ', np.min(y_source))

    print('================================')
    print('X_dev: ', x_dev.shape)
    print('Y_dev: ', y_dev.shape)
    print('Y_dev max: ', np.max(y_dev))
    print('Y_dev min: ', np.min(y_dev))

    print('================================')
    print('X_target: ', x_target.shape)
    print('Y_target: ', y_target.shape)
    print('Y_target max: ', np.max(y_target))
    print('Y_target min: ', np.min(y_target))
    print('================================')

    return {
        'x_source': x_source,
        'y_source': y_source,
        'source_prompts': source_prompts,
        'x_dev': x_dev,
        'y_dev': y_dev,
        'dev_prompts': dev_prompts,
        'x_target': x_target,
        'y_target': y_target,
        'target_prompts': target_prompts
    }


def load_data_PMAES(
    data_path: str,
    attribute_name: str,
    embedding_model: str,
    device: str,
    features_path: str = 'data/hand_crafted_v3.csv',
    readability_path: str = 'data/allreadability.pickle',
    devsize: int | float = 30
):
    _, _, test_data = create_embedding_features(
        data_path,
        attribute_name,
        embedding_model,
        device
    )
    _, _, _, _, dev_idx, test_idx = get_dev_sample(
        test_data['essay'],
        test_data['normalized_label'],
        dev_size=devsize
    )

    read_configs = {
        'train_path': data_path + 'train.pk',
        'dev_path': data_path + 'dev.pk',
        'test_path': data_path + 'test.pk',
        'features_path': features_path,
        'readability_path': readability_path
    }

    pos_vocab = read_pos_vocab(read_configs)
    train_data, valid_data, test_data = read_essays_single_score(read_configs, pos_vocab, attribute_name)

    max_sent_len = min(max(train_data['max_sentlen'], valid_data['max_sentlen'], test_data['max_sentlen']), 50)
    max_sent_num = min(max(train_data['max_sentnum'], valid_data['max_sentnum'], test_data['max_sentnum']), 100)
    print('max sent length: {}'.format(max_sent_len))
    print('max sent num: {}'.format(max_sent_num))
    train_data['score_scaled'] = get_single_scaled_down_score(train_data['data_y'], train_data['prompt_ids'], attribute_name)
    valid_data['score_scaled'] = get_single_scaled_down_score(valid_data['data_y'], valid_data['prompt_ids'], attribute_name)
    test_data['score_scaled'] = get_single_scaled_down_score(test_data['data_y'], test_data['prompt_ids'], attribute_name)

    train_prompt_ids = train_data['prompt_ids']
    dev_prompt_ids = valid_data['prompt_ids']
    test_prompt_ids = test_data['prompt_ids']
    train_essay_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sent_num, max_sent_len)
    valid_essay_pos = pad_hierarchical_text_sequences(valid_data['pos_x'], max_sent_num, max_sent_len)
    test_essay_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sent_num, max_sent_len)

    train_essay_pos = train_essay_pos.reshape((train_essay_pos.shape[0], train_essay_pos.shape[1] * train_essay_pos.shape[2]))
    valid_essay_pos = valid_essay_pos.reshape((valid_essay_pos.shape[0], valid_essay_pos.shape[1] * valid_essay_pos.shape[2]))
    test_essay_pos = test_essay_pos.reshape((test_essay_pos.shape[0], test_essay_pos.shape[1] * test_essay_pos.shape[2]))

    train_prompt_ids = np.concatenate([train_prompt_ids, dev_prompt_ids], axis=0)
    dev_prompt_ids = np.array(test_prompt_ids)[dev_idx]
    test_prompt_ids = np.array(test_prompt_ids)[test_idx]

    train_essay_pos = np.concatenate([train_essay_pos, valid_essay_pos], axis=0)
    valid_essay_pos = test_essay_pos[dev_idx]
    test_essay_pos = test_essay_pos[test_idx]

    train_score = np.concatenate([train_data['score_scaled'], valid_data['score_scaled']], axis=0)
    valid_score = np.array(test_data['score_scaled'])[dev_idx]
    test_score = np.array(test_data['score_scaled'])[test_idx]

    train_linguistic = np.concatenate([train_data['features_x'], valid_data['features_x']], axis=0)
    valid_linguistic = np.array(test_data['features_x'])[dev_idx]
    test_linguistic = np.array(test_data['features_x'])[test_idx]

    train_readability = np.concatenate([train_data['readability_x'], valid_data['readability_x']], axis=0)
    valid_readability = np.array(test_data['readability_x'])[dev_idx]
    test_readability = np.array(test_data['readability_x'])[test_idx]

    # print info
    print('================================')
    print('X_source: ', train_essay_pos.shape)
    print('X_source_linguistic_features: ', train_linguistic.shape)
    print('X_source_readability: ', train_readability.shape)
    print('Y_source: ', train_score.shape)

    print('================================')
    print('X_dev: ', valid_essay_pos.shape)
    print('X_dev_linguistic_features: ', valid_linguistic.shape)
    print('X_dev_readability: ', valid_readability.shape)
    print('Y_dev: ', valid_score.shape)

    print('================================')
    print('X_target: ', test_essay_pos.shape)
    print('X_target_linguistic_features: ', test_linguistic.shape)
    print('X_target_readability: ', test_readability.shape)
    print('Y_target: ', test_score.shape)
    print('================================')

    return {
        'pos_vocab': pos_vocab,
        'x_source': train_essay_pos,
        'x_source_linguistic_features': train_linguistic,
        'x_source_readability': train_readability,
        'source_prompts': train_prompt_ids,
        'y_source': train_score,
        'x_dev': valid_essay_pos,
        'x_dev_linguistic_features': valid_linguistic,
        'x_dev_readability': valid_readability,
        'dev_prompts': dev_prompt_ids,
        'y_dev': valid_score,
        'x_target': test_essay_pos,
        'x_target_linguistic_features': test_linguistic,
        'x_target_readability': test_readability,
        'target_prompts': test_prompt_ids,
        'y_target': test_score,
        'max_sentnum': max_sent_num,
        'max_sentlen': max_sent_len,
    }
