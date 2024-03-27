# %%
import numpy as np
import pandas as pd
import os
import random
import torch
from PAES.configs import PAESConfig
from dvrl.predictor_model import MLP
from transformers import AutoConfig
from utils.create_embedding_feautres import create_embedding_features
from utils.dvrl_utils import remove_top_p_sample, fit_func, pred_func, calc_qwk, random_remove_sample, get_dev_sample
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


for test_prompt_id in range(1, 9):
    print('Prompt: ', test_prompt_id)
    test_prompt_id = test_prompt_id
    output_path = f'outputs/DVRL_DomainAdaptation{test_prompt_id}_devsize30/'

    seed = 12
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    configs = PAESConfig()
    attribute_name = 'score'

    # Load data
    data_path = 'data/cross_prompt_attributes/' + str(test_prompt_id) + '/'
    model_name = 'microsoft/deberta-v3-large'

    ###################################################
    # Training MLP
    ###################################################
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights, model_name, device):
        config = AutoConfig.from_pretrained(model_name)
        pred_model = MLP(config.hidden_size)

        history = fit_func(pred_model, x_source, y_source, batch_size=256, epochs=100, device=device, sample_weight=weights)

        y_pred = pred_func(pred_model, x_test, batch_size=256, device=device)
        y_dev_pred = pred_func(pred_model, x_dev, batch_size=256, device=device)
        qwk = calc_qwk(y_test, y_pred, test_prompt_id, 'score')
        corr = np.corrcoef(y_test, np.array(y_pred).flatten())[0, 1]
        mse = mean_squared_error(y_test, y_pred)
        dev_mse = mean_squared_error(y_dev, y_dev_pred)
        print(f'QWK: {qwk:.4f}')
        return qwk, corr, mse, dev_mse

    def plot_results(p, high_qwks, low_qwks, random_qwks, high_corr, low_corr, random_corr, high_mse, low_mse, random_mse, high_loss, low_loss, output_path):
        plt.figure(figsize=(10, 5))
        plt.plot(p, high_qwks, label='Removing high value data', color='blue')
        plt.plot(p, low_qwks, label='Removing low value data', color='red')
        plt.plot(p, random_qwks, label='Randomly removing data', color='green')
        plt.title('Remove High/Low Value Samples: QWK')
        plt.xlabel('Fraction of Removed Samples')
        plt.ylabel('QWK')
        plt.legend()
        plt.savefig(output_path + 'remove_high_low_samples_qwk.png')

        plt.figure(figsize=(10, 5))
        plt.plot(p, high_corr, label='Removing high value data', color='blue')
        plt.plot(p, low_corr, label='Removing low value data', color='red')
        plt.plot(p, random_corr, label='Randomly removing data', color='green')
        plt.title('Remove High/Low Value Samples: Correlation')
        plt.xlabel('Fraction of Removed Samples')
        plt.ylabel('Correlation')
        plt.legend()
        plt.savefig(output_path + 'remove_high_low_samples_corr.png')

        plt.figure(figsize=(10, 5))
        plt.plot(p, high_mse, label='Removing high value data', color='blue')
        plt.plot(p, low_mse, label='Removing low value data', color='red')
        plt.plot(p, random_mse, label='Randomly removing data', color='green')
        plt.plot(p, low_loss, label='Dev Loss in removing low value data', color='red', linestyle='dashed')
        plt.plot(p, high_loss, label='Dev Loss in removing high value data', color='blue', linestyle='dashed')
        plt.title('Remove High/Low Value Samples: MSE')
        plt.xlabel('Fraction of Removed Samples')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(output_path + 'remove_high_low_samples_mse.png')

    set_seed(seed)
    train_data, val_data, test_data = create_embedding_features(data_path, attribute_name, model_name, device)
    x_source, y_source = np.concatenate([train_data['essay'], val_data['essay']]), np.concatenate([train_data['normalized_label'], val_data['normalized_label']])
    x_dev, x_test, y_dev, y_test, dev_ids, _ = get_dev_sample(test_data['essay'], test_data['normalized_label'], dev_size=30)

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

    interval = 0.1
    p = np.arange(0.0, 1.0, interval)
    high_qwks, high_corr, high_mse, high_loss = [], [], [], []
    low_qwks, low_corr, low_mse, low_loss = [], [], [], []
    random_qwks, random_corr, random_mse = [], [], []

    for p_val in p:
        set_seed(seed)
        weights = remove_top_p_sample(np.load(output_path + 'estimated_data_value.npy'), top_p=p_val, ascending=False)
        qwk, corr, mse, dev_mse = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights, model_name, device)
        high_qwks.append(qwk)
        high_corr.append(corr)
        high_mse.append(mse)
        high_loss.append(dev_mse)

        set_seed(seed)
        weights = remove_top_p_sample(np.load(output_path + 'estimated_data_value.npy'), top_p=p_val, ascending=True)
        qwk, corr, mse, dev_mse = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights, model_name, device)
        low_qwks.append(qwk)
        low_corr.append(corr)
        low_mse.append(mse)
        low_loss.append(dev_mse)

        set_seed(seed)
        weights = random_remove_sample(np.load(output_path + 'estimated_data_value.npy'), remove_p=p_val)
        qwk, corr, mse, _ = train_and_evaluate(x_source, y_source, x_dev, y_dev, x_test, y_test, test_prompt_id, weights, model_name, device)
        random_qwks.append(qwk)
        random_corr.append(corr)
        random_mse.append(mse)

    plot_results(p, high_qwks, low_qwks, random_qwks, high_corr, low_corr, random_corr, high_mse, low_mse, random_mse, high_loss, low_loss, output_path)

    output_qwk = np.array([p.tolist(), high_qwks, low_qwks, random_qwks])
    output_corr = np.array([p.tolist(), high_corr, low_corr, random_corr])
    output_mse = np.array([p.tolist(), high_mse, low_mse, random_mse])
    pd.DataFrame(output_qwk).to_csv(output_path + 'mlp_qwk.csv', index=False, header=False)
    pd.DataFrame(output_corr).to_csv(output_path + 'mlp_corr.csv', index=False, header=False)
    pd.DataFrame(output_mse).to_csv(output_path + 'mlp_mse.csv', index=False, header=False)

    high_min_index = np.array(high_loss).argmin()
    low_min_index = np.array(low_loss).argmin()
    print('QWK-high: ', high_qwks[high_min_index])
    print('Correlation-high: ', high_corr[high_min_index])
    print('QWK-low: ', low_qwks[low_min_index])
    print('Correlation-low: ', low_corr[low_min_index])

    # 指標をデータフレームに格納
    metrics_df = pd.DataFrame({
        'Metric': ['QWK-high', 'Correlation-high', 'QWK-low', 'Correlation-low'],
        'Value': [high_qwks[high_min_index], high_corr[high_min_index], low_qwks[low_min_index], low_corr[low_min_index]]
    })

    # CSVファイルに保存
    metrics_df.to_csv(output_path + 'metrics_summary.csv', index=False)

    # ###################################################
    # # Train BERT by development set
    # ###################################################
    # from utils.create_embedding_feautres import load_data, normalize_scores, create_data_loader
    # from transformers import AutoTokenizer, AutoModel, AutoConfig

    # data = load_data(data_path)

    # features = np.array(data['test']['feature'])
    # labels = np.array(data['test']['label'])
    # prompts = np.array(data['test']['essay_set'])
    # ids = np.array(data['test']['essay_id'])
    # # Normalize scores
    # normalized_labels = normalize_scores(labels, prompts, attribute_name)


    # sample_id = np.load(output_path + 'dev_ids.npy')
    # not_sample_id = np.array([i for i in range(len(features)) if i not in sample_id])

    # train_data = {}
    # test_data = {}

    # train_data['feature'] = features[sample_id]
    # train_data['normalized_label'] = normalized_labels[sample_id]
    # train_data['essay_set'] = prompts[sample_id]

    # test_data['feature'] = features[not_sample_id]
    # test_data['normalized_label'] = normalized_labels[not_sample_id]
    # test_data['essay_set'] = prompts[not_sample_id]

    # print(train_data['feature'].shape)
    # print(test_data['feature'].shape)

    # import torch
    # import torch.nn as nn
    # from torch.optim import AdamW
    # from transformers import get_linear_schedule_with_warmup
    # from utils.evaluation import train_epoch, evaluate_epoch
    # from models.AES import BERT_Regressor


    # model_name = 'bert-base-uncased'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name).to(device)
    # config = AutoConfig.from_pretrained(model_name)

    # train_loader = create_data_loader(train_data, tokenizer, max_length=512, batch_size=32)
    # test_loader = create_data_loader(test_data, tokenizer, max_length=512, batch_size=32)

    # # set parameters
    # EPOCHS = 30

    # # Define the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device}")

    # # Initialize the model
    # model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)

    # # Define loss function, optimizer, and scheduler
    # loss_fn = nn.MSELoss(reduction='none').to(device)
    # optimizer = AdamW(model.parameters(), lr=2e-5)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

    # # Training loop
    # for epoch in range(EPOCHS):
    #     print('valuationに使用したデータだけで訓練中')
    #     print(f"Epoch {epoch+1}/{EPOCHS}")

    #     # Training Set
    #     train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
    #     print(f"Training loss: {train_loss}")

    #     # Test Set
    #     eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
    #     print(f"Test loss: {eval_history['loss']:.4f}")

    #     print(f'[TEST] -> QWK: {eval_history["qwk"]: .3f}, CORR: {eval_history["corr"]: .3f}, RMSE: {eval_history["rmse"]: .3f}')

    # best_val_metrics = [eval_history[met] for met in ['qwk', 'lwk', 'corr', 'rmse', 'mae']]

    # pd.DataFrame(np.array(best_val_metrics).reshape(1, 5), columns=['qwk', 'lwk', 'corr', 'rmse', 'mae']).to_csv(output_path + f'BERT-onlydev{test_prompt_id}.csv', index=False, header=True)


    ##################################################
    # Train BERT by valuable data.
    ##################################################
    low_mean_mse = np.array(low_mse).mean()
    high_mean_mse = np.array(high_mse).mean()

    if low_mean_mse > high_mean_mse:
        逆張り = True
    else:
        逆張り = False

    from utils.create_embedding_feautres import load_data, normalize_scores, create_data_loader
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    from utils.evaluation import train_epoch, evaluate_epoch
    from models.transfomer_enc import BERT_Regressor

    qwks = []
    corr = []
    dev_loss = []
    interval = 0.1
    for p in np.arange(0.0, 1.0, interval):
        data_values = np.load(output_path + 'estimated_data_value.npy')
        if 逆張り:
            weights = remove_top_p_sample(data_values, top_p=p, ascending=False)
        else:
            weights = remove_top_p_sample(data_values, top_p=p, ascending=True)
        use_train_sample = (weights == 1)
        
        # Load data
        data = load_data(data_path)
        
        x_train = np.array(data['train']['feature'])
        x_dev = np.array(data['dev']['feature'])
        x_test = np.array(data['test']['feature'])
        
        y_train = np.array(data['train']['label'])
        y_dev = np.array(data['dev']['label'])
        y_test = np.array(data['test']['label'])
        
        train_essay_prompt = np.array(data['train']['essay_set'])
        dev_essay_prompt = np.array(data['dev']['essay_set'])
        test_essay_prompt = np.array(data['test']['essay_set'])
        
        train_essay_id = np.array(data['train']['essay_id'])
        dev_essay_id = np.array(data['dev']['essay_id'])
        test_essay_id = np.array(data['test']['essay_id'])
        
        # Normalize scores
        y_train = normalize_scores(y_train, train_essay_prompt, attribute_name)
        y_dev = normalize_scores(y_dev, dev_essay_prompt, attribute_name)
        y_test = normalize_scores(y_test, test_essay_prompt, attribute_name)
        
        sample_id = np.load(output_path + 'dev_ids.npy')
        not_sample_id = np.array([i for i in range(len(y_test)) if i not in sample_id])
        
        train_data = {}
        dev_data = {}
        test_data = {}
        
        train_data['feature'] = np.concatenate([x_train, x_dev], axis=0)[use_train_sample]
        train_data['normalized_label'] = np.concatenate([y_train, y_dev], axis=0)[use_train_sample]
        train_data['essay_set'] = np.concatenate([train_essay_prompt, dev_essay_prompt], axis=0)
        
        dev_data['feature'] = x_test[sample_id]
        dev_data['normalized_label'] = y_test[sample_id]
        dev_data['essay_set'] = test_essay_prompt[sample_id]
        
        test_data['feature'] = x_test[not_sample_id]
        test_data['normalized_label'] = y_test[not_sample_id]
        test_data['essay_set'] = test_essay_prompt[not_sample_id]
        
        print(f'train size: {train_data["feature"].shape}')
        print(f'dev size: {dev_data["feature"].shape}')
        print(f'test size: {test_data["feature"].shape}')
        
        
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        config = AutoConfig.from_pretrained(model_name)
        

        train_loader = create_data_loader(train_data, tokenizer, max_length=512, batch_size=16)
        dev_loader = create_data_loader(dev_data, tokenizer, max_length=512, batch_size=16)
        test_loader = create_data_loader(test_data, tokenizer, max_length=512, batch_size=16)
        
        # set parameters
        EPOCHS = 10
        
        # Define the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        
        # Initialize the model
        model = BERT_Regressor(model, hidden_size=config.hidden_size).to(device)
        
        # Define loss function, optimizer, and scheduler
        loss_fn = nn.MSELoss(reduction='none').to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)
        
        # Training loop
        best_test_metrics = [-1, -1, -1, -1, -1]
        best_val_metrics = [-1, -1, -1, -1, -1]
        best_dev_loss = 1000
        for epoch in range(EPOCHS):
            print('BERT訓練中')
            print(f"Epoch {epoch+1}/{EPOCHS}")
        
            # Training Set
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, use_weight=False)
            print(f"Training loss: {train_loss}")
        
            # Development Set
            dev_history = evaluate_epoch(model, dev_loader, loss_fn, device, attribute_name)
            print(f"Dev loss: {dev_history['loss']:.4f}")
        
            # Test Set
            eval_history = evaluate_epoch(model, test_loader, loss_fn, device, attribute_name)
            print(f"Test loss: {eval_history['loss']:.4f}")
        
            print(f'[VAL]  -> QWK: {dev_history["qwk"]: .3f}, CORR: {dev_history["corr"]: .3f}, RMSE: {dev_history["rmse"]: .3f}')
            print(f'[TEST] -> QWK: {eval_history["qwk"]: .3f}, CORR: {eval_history["corr"]: .3f}, RMSE: {eval_history["rmse"]: .3f}')
        
            if dev_history["qwk"] > best_val_metrics[0]:
                best_dev_loss = dev_history['loss']
                for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
                    best_val_metrics[i] = dev_history[met]
                    best_test_metrics[i] = eval_history[met]
        
            print(f'[BEST] -> QWK: {best_test_metrics[0]: .3f}, CORR: {best_test_metrics[2]: .3f}, RMSE: {best_test_metrics[3]: .3f}')
        

        qwks.append(best_test_metrics[0])
        corr.append(best_test_metrics[2])
        dev_loss.append(best_dev_loss)

    output_metrics = np.array([np.arange(0.0, 1.0, interval).tolist(), qwks, corr, dev_loss])
    pd.DataFrame(output_metrics).to_csv(output_path + 'DVRL-BERT.csv', index=False, header=False)

    min_index = np.array(dev_loss).argmin()
    print('QWK: ', qwks[min_index])
    print('Correlation: ', corr[min_index])


    # ##################################################
    # # Train PAES by valuable data
    # ##################################################

    # qwks = []
    # corr = []
    # dev_loss = []
    # interval = 0.1
    # for p in np.arange(0.0, 1.0, interval):
    #     from utils.read_data import read_essays_single_score, read_pos_vocab
    #     from utils.general_utils import get_single_scaled_down_score, pad_hierarchical_text_sequences
    #     from torch.utils.data import TensorDataset, DataLoader
    #     from PAES.models import PAES, fastPAES
    #     from utils.evaluation import train_model, evaluate_model
        
    #     # Load configs
    #     configs = PAESConfig()
        
    #     data_path = configs.DATA_PATH3
    #     print(f'load data from {data_path}...')
    #     train_path = data_path + str(test_prompt_id) + '/train.pk'
    #     dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    #     test_path = data_path + str(test_prompt_id) + '/test.pk'
    #     features_path = configs.FEATURES_PATH
    #     readability_path = configs.READABILITY_PATH
    #     epochs = configs.EPOCHS
    #     batch_size = configs.BATCH_SIZE
        
    #     read_configs = {
    #         'train_path': train_path,
    #         'dev_path': dev_path,
    #         'test_path': test_path,
    #         'features_path': features_path,
    #         'readability_path': readability_path
    #     }
        
    #     # Read data
    #     pos_vocab = read_pos_vocab(read_configs)
    #     train_data, dev_data, test_data = read_essays_single_score(read_configs, pos_vocab, attribute_name)
        
    #     # Get max sentence length and max sentence number
    #     max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    #     max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])
        
    #     # Scale down the scores
    #     train_data['y_scaled'] = get_single_scaled_down_score(train_data['data_y'], train_data['prompt_ids'], attribute_name)
    #     dev_data['y_scaled'] = get_single_scaled_down_score(dev_data['data_y'], dev_data['prompt_ids'], attribute_name)
    #     test_data['y_scaled'] = get_single_scaled_down_score(test_data['data_y'], test_data['prompt_ids'], attribute_name)
        
    #     # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
    #     X_train_pos = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
    #     X_dev_pos = pad_hierarchical_text_sequences(dev_data['pos_x'], max_sentnum, max_sentlen)
    #     X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)
        
    #     X_train_pos = X_train_pos.reshape((X_train_pos.shape[0], X_train_pos.shape[1] * X_train_pos.shape[2]))
    #     X_dev_pos = X_dev_pos.reshape((X_dev_pos.shape[0], X_dev_pos.shape[1] * X_dev_pos.shape[2]))
    #     X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))
        
    #     # convert to tensor
    #     X_train= torch.tensor(X_train_pos, dtype=torch.long)
    #     X_dev = torch.tensor(X_dev_pos, dtype=torch.long)
    #     X_test= torch.tensor(X_test_pos, dtype=torch.long)
        
    #     X_train_linguistic_features = torch.tensor(np.array(train_data['features_x']), dtype=torch.float)
    #     X_dev_linguistic_features = torch.tensor(np.array(dev_data['features_x']), dtype=torch.float)
    #     X_test_linguistic_features = torch.tensor(np.array(test_data['features_x']), dtype=torch.float)
        
    #     X_train_readability = torch.tensor(np.array(train_data['readability_x']), dtype=torch.float)
    #     X_dev_readability = torch.tensor(np.array(dev_data['readability_x']), dtype=torch.float)
    #     X_test_readability = torch.tensor(np.array(test_data['readability_x']), dtype=torch.float)
        
    #     Y_train = torch.tensor(np.array(train_data['y_scaled']), dtype=torch.float)
    #     Y_dev = torch.tensor(np.array(dev_data['y_scaled']), dtype=torch.float)
    #     Y_test = torch.tensor(np.array(test_data['y_scaled']), dtype=torch.float)
        
    #     train_essay_set = torch.tensor(np.array(train_data['prompt_ids']), dtype=torch.long)
    #     dev_essay_set = torch.tensor(np.array(dev_data['prompt_ids']), dtype=torch.long)
    #     test_essay_set = torch.tensor(np.array(test_data['prompt_ids']), dtype=torch.long)
        
    #     # Load weights
    #     data_values = np.load(output_path + 'estimated_data_value.npy')
    #     if 逆張り:
    #         weights = remove_top_p_sample(data_values, top_p=p, ascending=False)
    #     else:
    #         weights = remove_top_p_sample(data_values, top_p=p, ascending=True)
    #     weights = (torch.tensor(weights, dtype=torch.float) == 1)
        
    #     sample_id = np.load(output_path + 'dev_ids.npy')
    #     not_sample_id = np.array([i for i in range(len(y_test)) if i not in sample_id])
        
    #     X_train = torch.concat([X_train, X_dev])[weights]
    #     Y_train = torch.concat([Y_train, Y_dev])[weights]
    #     X_train_linguistic_features = torch.concat([X_train_linguistic_features, X_dev_linguistic_features])[weights]
    #     X_train_readability = torch.concat([X_train_readability, X_dev_readability])[weights]
    #     train_essay_set = torch.concat([train_essay_set, dev_essay_set])[weights]

    #     X_dev = X_test[sample_id]
    #     Y_dev = Y_test[sample_id]
    #     X_dev_linguistic_features = X_test_linguistic_features[sample_id]
    #     X_dev_readability = X_test_readability[sample_id]
    #     dev_essay_set = test_essay_set[sample_id]
        
    #     X_test = X_test[not_sample_id]
    #     Y_test = Y_test[not_sample_id]
    #     X_test_linguistic_features = X_test_linguistic_features[not_sample_id]
    #     X_test_readability = X_test_readability[not_sample_id]
    #     test_essay_set = test_essay_set[not_sample_id]
        
    #     print(f'train size: {X_train.size()}')
    #     print(f'dev size: {X_dev.size()}')
    #     print(f'test size: {X_test.size()}')
        
    #     # Create Datasets
    #     train_dataset = TensorDataset(X_train, Y_train, X_train_linguistic_features, X_train_readability, train_essay_set)
    #     dev_dataset = TensorDataset(X_dev, Y_dev, X_dev_linguistic_features, X_dev_readability, dev_essay_set)
    #     test_dataset = TensorDataset(X_test, Y_test, X_test_linguistic_features, X_test_readability, test_essay_set)
    #     # Create Dataloaders
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    #     dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        
    #     import torch.nn as nn
    #     import torch.optim as optim
    #     import torch
        
    #     model = fastPAES(max_sentnum, max_sentlen, X_train_linguistic_features.size(1), X_train_readability.size(1), pos_vocab=pos_vocab)
    #     model = model.to(device)
    #     print(model)
        
    #     # Create loss and optimizer
    #     reduction = 'mean'
    #     MSE_Loss = nn.MSELoss(reduction=reduction).to(device)
    #     optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        
    #     train_history = []
    #     dev_history = []
    #     test_history = []
    #     best_test_metrics = [-1, -1, -1, -1, -1]
    #     best_val_metrics = [-1, -1, -1, -1, -1]
    #     best_dev_loss = 1000
    #     for epoch in range(epochs):
    #         print('PAES訓練中')
    #         print('{} / {} EPOCHS'.format(epoch+1, epochs))
    #         print('Seed: {}, Prompt: {}'.format(seed, test_prompt_id))
            
    #         # Train the model
    #         train_loss = train_model(model, train_loader, MSE_Loss, optimizer, device, weight=False)
    #         print(f'Train loss: {train_loss: .4f}')
    #         train_history.append(train_loss)
        
    #         # Evaluate the model on dev set
    #         dev_results = evaluate_model(model, dev_loader, MSE_Loss, device, attribute_name)
    #         print(f'Validation loss: {dev_results["loss"]: .4f}')
    #         dev_history.append(dev_results["loss"])
        
    #         # Evaluate the model on test set
    #         test_results = evaluate_model(model, test_loader, MSE_Loss, device, attribute_name)
    #         print(f'Test loss: {test_results["loss"]: .4f}')
    #         test_history.append(test_results["loss"])
        
    #         if dev_results["qwk"] > best_val_metrics[0]:
    #             best_dev_loss = dev_results['loss']
    #             for i, met in enumerate(['qwk', 'lwk', 'corr', 'rmse', 'mae']):
    #                 best_val_metrics[i] = dev_results[met]
    #                 best_test_metrics[i] = test_results[met]
        
    #         if epoch % 10 == 0:
    #             print(f'[VAL]  -> QWK: {dev_results["qwk"]: .3f}, CORR: {dev_results["corr"]: .3f}, RMSE: {dev_results["rmse"]: .3f}')
    #             print(f'[TEST] -> QWK: {test_results["qwk"]: .3f}, CORR: {test_results["corr"]: .3f}, RMSE: {test_results["rmse"]: .3f}')
    #             print(f'[BEST] -> QWK: {best_test_metrics[0]: .3f}, CORR: {best_test_metrics[2]: .3f}, RMSE: {best_test_metrics[3]: .3f}')
        
    #     qwks.append(best_test_metrics[0])
    #     corr.append(best_test_metrics[2])
    #     dev_loss.append(best_dev_loss)

    # output_qwk = np.array([np.arange(0.0, 1.0, interval).tolist(), qwks, corr, dev_loss])
    # pd.DataFrame(output_qwk).to_csv(output_path + 'DVRL-PAES.csv', index=False, header=False)


