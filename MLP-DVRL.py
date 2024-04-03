import argparse
import numpy as np
from transformers import AutoConfig
import wandb

from utils.general_utils import set_seed
from utils.create_embedding_feautres import create_embedding_features
from utils.dvrl_utils import remove_top_p_sample, fit_func, pred_func, calc_qwk, random_remove_sample, get_dev_sample
from dvrl.predictor_model import MLP
from sklearn.metrics import mean_squared_error


def main(args):
    test_prompt_id = args.test_prompt_id
    data_value_path = f'outputs/DVRL_DomainAdaptation{test_prompt_id}_devsize30/'
    seed = args.seed
    set_seed(seed)
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    attribute_name = args.attribute_name
    data_path = args.data_dir + str(test_prompt_id) + '/'
    model_name = args.embedding_model

    ###################################################
    # Training MLP
    ###################################################
    def train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights):
        config = AutoConfig.from_pretrained(model_name)
        pred_model = MLP(config.hidden_size)

        fit_func(pred_model, x_source, y_source, batch_size=batch_size, epochs=epochs, device=device, sample_weight=weights)

        y_pred = pred_func(pred_model, x_test, batch_size=batch_size, device=device)
        y_dev_pred = pred_func(pred_model, x_dev, batch_size=batch_size, device=device)
        qwk = calc_qwk(y_test, y_pred, test_prompt_id, attribute_name)
        corr = np.corrcoef(y_test, np.array(y_pred).flatten())[0, 1]
        mse = mean_squared_error(y_test, y_pred)
        dev_mse = mean_squared_error(y_dev, y_dev_pred)
        return qwk, corr, mse, dev_mse

    # Load data
    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    x_source, y_source = np.concatenate([train_data['essay'], val_data['essay']]), np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    x_dev, x_test, y_dev, y_test, _, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=args.dev_size)

    # use dev data to train
    x_source, y_source = np.concatenate([x_source, x_dev]), np.concatenate([y_source, y_dev])

    print('================================')
    print('X_train: ', x_source.shape)
    print('Y_train: ', y_source.shape)
    print('Y_train max: ', np.max(y_source))
    print('Y_train min: ', np.min(y_source))

    print('================================')
    print('X_dev: ', x_dev.shape)
    print('Y_dev: ', y_dev.shape)
    print('Y_dev max: ', np.max(y_dev))
    print('Y_dev min: ', np.min(y_dev))

    print('================================')
    print('X_test: ', y_test.shape)
    print('Y_test: ', y_test.shape)
    print('Y_test max: ', np.max(y_test))
    print('Y_test min: ', np.min(y_test))
    print('================================')

    wandb.init(project=args.pj_name, name=args.run_name+str(test_prompt_id), config=args)
    interval = 0.1
    p = np.arange(0.0, 1.0, interval)
    for p_val in p:
        # データの価値が低いものを削除
        set_seed(seed)
        weights = remove_top_p_sample(np.load(data_value_path + 'estimated_data_value.npy'), top_p=p_val, ascending=False)
        weights = np.concatenate([weights, np.array([1]*x_dev.shape[0])])
        qwk_high, _, _, dev_loss_high = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights)

        # データの価値が高いものを削除
        set_seed(seed)
        weights = remove_top_p_sample(np.load(data_value_path + 'estimated_data_value.npy'), top_p=p_val, ascending=True)
        weights = np.concatenate([weights, np.array([1]*x_dev.shape[0])])
        qwk_low, _, _, dev_loss_low = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights)

        # データをランダムに削除
        set_seed(seed)
        weights = random_remove_sample(np.load(data_value_path + 'estimated_data_value.npy'), remove_p=p_val)
        weights = np.concatenate([weights, np.array([1]*x_dev.shape[0])])
        qwk_random, _, _, dev_loss_random = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights)

        wandb.log({
            'p': p_val,
            'QWK[High]': qwk_high,
            'QWK[Low]': qwk_low,
            'QWK[Random]': qwk_random,
            'Dev Loss[High]': dev_loss_high,
            'Dev Loss[Low]': dev_loss_low,
            'Dev Loss[Random]': dev_loss_random
        })

    wandb.finish()


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pj_name', type=str, default='DVRL', help='wandb project name for logging')
    parser.add_argument('--run_name', type=str, default='MLP-DVRL', help='name of the experiment')
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--attribute_name', type=str, default='score', help='name of the attribute to be trained on')
    parser.add_argument('--data_dir', type=str, default='data/cross_prompt_attributes/', help='data directory')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv', help='path to hand crafted features')
    parser.add_argument('--readability_path', type=str, default='data/allreadability.pickle', help='path to readability features')
    parser.add_argument('--embedding_model', type=str, default='microsoft/deberta-v3-large', help='name of the embedding model')
    parser.add_argument('--dev_size', type=int, default=30, help='size of development set')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    
    args = parser.parse_args()
    print(dict(args._get_kwargs()))

    main(args)