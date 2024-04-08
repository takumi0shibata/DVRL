import os
import time
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import wandb

from models.PMAES import EssayEncoder, Scorer, PromptMappingCL
from utils.read_data import read_pos_vocab, read_essays_single_score
from utils.general_utils import pad_hierarchical_text_sequences, get_single_scaled_down_score

import torch
from utils.pmaes_utils import PMAESDataSet, TrainSingleOverallScoring
from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences
from utils.create_embedding_feautres import create_embedding_features
from utils.dvrl_utils import get_dev_sample


def seed_all(seed_value):
    """
    Setting the random seed across the pipeline
    """
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value)
    np.random.seed(seed_value) # cpu vars
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PAES_attributes models")
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='PMAES-FullSource-nodev', help='name of the experiment')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--target_id', type=int, default=1, help='set random seed')
    parser.add_argument('--attribute_name', type=str, default='score', help='set random seed')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--source2target', type=str, default='many2one', help='Setting of source-target pair')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Only useful when embedding is randomly initialised')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--filter_num', type=int, default=100, help='Num of filters in conv layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='filter length in 1st conv layer')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='Recurrent type')
    parser.add_argument('--lstm_units', type=int, default=50, help='Num of hidden units in recurrent layer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for layers')
    parser.add_argument('--device', type=str, help='cpu or gpu', default='cuda')
    parser.add_argument('--batch_num', type=int, default=20, help='Number of batches')
    parser.add_argument('--max_sentlen', type=int, default=50, help='Max sentence length')
    parser.add_argument('--max_sentnum', type=int, default=100, help='Max sentence number')

    args = parser.parse_args()
    test_prompt_id = args.target_id
    seed = args.seed

    seed_all(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))

    data_path = args.data_dir
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    features_path = args.features_path
    readability_path = args.readability_path

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
    }

    ########################################################
    # get dev and test indices
    _, _, test_data = create_embedding_features(args.data_dir + str(test_prompt_id) + '/', 'score', args.embedding_model, args.device)
    _, _, _, _, dev_idx, test_idx = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)
    ########################################################

    pos_vocab = read_pos_vocab(read_configs)
    train_data, valid_data, test_data = read_essays_single_score(read_configs, pos_vocab, args.attribute_name)

    max_sent_len = min(max(train_data['max_sentlen'], valid_data['max_sentlen'], test_data['max_sentlen']), args.max_sentlen)
    max_sent_num = min(max(train_data['max_sentnum'], valid_data['max_sentnum'], test_data['max_sentnum']), args.max_sentnum)
    print('max sent length: {}'.format(max_sent_len))
    print('max sent num: {}'.format(max_sent_num))
    train_data['score_scaled'] = get_single_scaled_down_score(train_data['data_y'], train_data['prompt_ids'], args.attribute_name)
    valid_data['score_scaled'] = get_single_scaled_down_score(valid_data['data_y'], valid_data['prompt_ids'], args.attribute_name)
    test_data['score_scaled'] = get_single_scaled_down_score(test_data['data_y'], test_data['prompt_ids'], args.attribute_name)

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

    tr_s_num, tr_t_num = len(train_essay_pos), len(test_essay_pos)
    batch_num = args.batch_num
    s_batch_size = int(tr_s_num / batch_num)
    t_batch_size = int(tr_t_num / batch_num)
    s_batch_num = int(tr_s_num / s_batch_size)
    t_batch_num = int(tr_t_num / t_batch_size)
    while t_batch_num > s_batch_num:
        s_batch_size -= 1
        s_batch_num = int(tr_s_num / s_batch_size)

    print('s_batch_size: {}'.format(s_batch_size))
    print('t_batch_size: {}'.format(t_batch_size))
    tr_s_loader = DataLoader(PMAESDataSet(train_prompt_ids, train_essay_pos, train_linguistic, train_readability, train_score), batch_size=s_batch_size)
    va_s_loader = DataLoader(PMAESDataSet(dev_prompt_ids, valid_essay_pos, valid_linguistic, valid_readability, valid_score), batch_size=s_batch_size)
    te_t_loader = DataLoader(PMAESDataSet(test_prompt_ids, test_essay_pos, test_linguistic, test_readability, test_score), batch_size=t_batch_size)

    essay_encoder = EssayEncoder(args, max_num=max_sent_num, max_len=max_sent_len, embed_dim=args.embedding_dim, pos_vocab=pos_vocab).to(args.device)
    scorer = Scorer(args).to(args.device)
    pm_cl = PromptMappingCL(args, tr_s_num, tr_t_num).to(args.device)
    optims = torch.optim.Adam([{'params': essay_encoder.parameters()}, {'params': scorer.parameters()}, {'params': pm_cl.parameters()}], lr=args.learning_rate)
    tr_log = {
        'Epoch_best_dev_qwk': [0, 0, 0],
        'Best_dev_qwk': [0, 0],
    }
    epochs = args.num_epochs
    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)
    for e_index in range(1, epochs+1):

        va_loss, te_loss, va_qwk, te_qwk, best_qwk = TrainSingleOverallScoring(args,
                                                                                essay_encoder, scorer, pm_cl, optims,
                                                                                tr_s_loader, va_s_loader, te_t_loader,
                                                                                args.target_id, e_index,
                                                                                tr_log, args.attribute_name
                                                                                )

        wandb.log({
            'dev_loss': va_loss,
            'test_loss': te_loss,
            'dev_qwk': va_qwk,
            'test_qwk': te_qwk,
            'best_qwk': best_qwk
        })

    wandb.finish()