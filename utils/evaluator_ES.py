import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.general_utils import get_min_max_score_vector


class evaluator():
    def __init__(self, num_item, overall_range, item_mask, analytic_range, dev_features_list, test_features_list, prompt_id):
        self.prompt_id = prompt_id
        self.item_mask = item_mask
        self.overall_range = overall_range
        self.analytic_range = analytic_range
        self.num_item = num_item
        self.dev_features_list = dev_features_list
        self.test_features_list = test_features_list
        self.dev_qwk_mean_best = -1
        self.dev_qwk = None
        self.dev_lwk = None
        self.dev_rmse = None
        self.dev_mae = None
        self.dev_corr = None
        self.test_qwk = None
        self.test_lwk = None
        self.test_rmse = None
        self.test_mae = None
        self.test_corr = None
        self.best_qwk_mean = -1
        self.best_qwk = None
        self.best_lwk = None
        self.best_rmse = None
        self.best_mae = None
        self.best_corr = None
        self.best_epoch = -1


    def calc_qwk(self, y_true, y_pred):
        qwk_scores = []
        for i in range(self.num_item):
            if i == 0:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='quadratic', labels=[i for i in range(self.overall_range)])
                qwk_scores.append(kappa_score)
            else:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='quadratic', labels=[i for i in range(self.analytic_range)])
                qwk_scores.append(kappa_score)
        return np.array(qwk_scores)


    def calc_lwk(self, y_true, y_pred):
        lwk_scores = []
        for i in range(self.num_item):
            if i == 0:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='linear', labels=[i for i in range(self.overall_range)])
                lwk_scores.append(kappa_score)
            else:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='linear', labels=[i for i in range(self.analytic_range)])
                lwk_scores.append(kappa_score)
        return np.array(lwk_scores)


    def calc_rmse(self, y_true, y_pred):
        rmse_scores = []
        for i in range(self.num_item):
            rmse_score = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            rmse_scores.append(rmse_score)
        return np.array(rmse_scores)


    def calc_mae(self, y_true, y_pred):
        mae_scores = []
        for i in range(self.num_item):
            mae_score = mean_absolute_error(y_true[:, i], y_pred[:, i])
            mae_scores.append(mae_score)
        return np.array(mae_scores)


    def calc_corr(self, y_true, y_pred):
        corr_scores = []
        for i in range(self.num_item):
            corr_score = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
            corr_scores.append(corr_score)
        return np.array(corr_scores)


    def evaluate_from_reg(self, model, y_true_org_dev, y_true_org_test, epoch):
        # Calculate score range
        min_score = np.array(get_min_max_score_vector()[self.prompt_id]['min'])[self.item_mask]
        max_score = np.array(get_min_max_score_vector()[self.prompt_id]['max'])[self.item_mask]
        score_range = max_score - min_score

        # Predict scores
        y_pred_dev = model.predict(self.dev_features_list)
        y_pred_test = model.predict(self.test_features_list)
        y_pred_dev = y_pred_dev * score_range
        y_pred_test = y_pred_test * score_range

        y_pred_dev = np.round(y_pred_dev)
        y_pred_test = np.round(y_pred_test)

        # Set the minimum score to 0
        y_true_dev = y_true_org_dev - min_score
        y_true_test = y_true_org_test - min_score

        # Calculate metrics
        self.dev_qwk = self.calc_qwk(y_true_dev, y_pred_dev)
        self.dev_lwk = self.calc_lwk(y_true_dev, y_pred_dev)
        self.dev_rmse = self.calc_rmse(y_true_dev, y_pred_dev)
        self.dev_mae = self.calc_mae(y_true_dev, y_pred_dev)
        self.dev_corr = self.calc_corr(y_true_dev, y_pred_dev)

        self.test_qwk = self.calc_qwk(y_true_test, y_pred_test)
        self.test_lwk = self.calc_lwk(y_true_test, y_pred_test)
        self.test_rmse = self.calc_rmse(y_true_test, y_pred_test)
        self.test_mae = self.calc_mae(y_true_test, y_pred_test)
        self.test_corr = self.calc_corr(y_true_test, y_pred_test)

        if np.mean(self.dev_qwk) > self.dev_qwk_mean_best:
            self.dev_qwk_mean_best = np.mean(self.dev_qwk)
            self.best_qwk = self.test_qwk
            self.best_lwk = self.test_lwk
            self.best_rmse = self.test_rmse
            self.best_mae = self.test_mae
            self.best_corr = self.test_corr
            self.best_epoch = epoch


    def evaluate_from_prob(self, model, y_true_org_dev, y_true_org_test, min_score, epoch, predict_option='ex'):
        # Predict scores
        y_pred_dev = model.predict(self.dev_features_list)
        y_pred_test = model.predict(self.test_features_list)
        if predict_option == 'ex':
            y_pred_dev = np.sum(y_pred_dev * np.arange(0, self.overall_range), axis=-1) # expected score
            y_pred_dev = np.round(y_pred_dev)
            y_pred_test = np.sum(y_pred_test * np.arange(0, self.overall_range), axis=-1) # expected score
            y_pred_test = np.round(y_pred_test)
        elif predict_option == 'argmax':
            y_pred_dev = np.argmax(y_pred_dev, axis=-1)
            y_pred_test = np.argmax(y_pred_test, axis=-1)

        # Set the minimum score to 0
        y_true_dev = y_true_org_dev - min_score
        y_true_test = y_true_org_test - min_score

        # Calculate metrics
        self.dev_qwk = self.calc_qwk(y_true_dev, y_pred_dev)
        self.dev_lwk = self.calc_lwk(y_true_dev, y_pred_dev)
        self.dev_rmse = self.calc_rmse(y_true_dev, y_pred_dev)
        self.dev_mae = self.calc_mae(y_true_dev, y_pred_dev)
        self.dev_corr = self.calc_corr(y_true_dev, y_pred_dev)

        self.test_qwk = self.calc_qwk(y_true_test, y_pred_test)
        self.test_lwk = self.calc_lwk(y_true_test, y_pred_test)
        self.test_rmse = self.calc_rmse(y_true_test, y_pred_test)
        self.test_mae = self.calc_mae(y_true_test, y_pred_test)
        self.test_corr = self.calc_corr(y_true_test, y_pred_test)

        if np.mean(self.dev_qwk) > self.dev_qwk_mean_best:
            self.dev_qwk_mean_best = np.mean(self.dev_qwk)
            self.best_qwk = self.test_qwk
            self.best_lwk = self.test_lwk
            self.best_rmse = self.test_rmse
            self.best_mae = self.test_mae
            self.best_corr = self.test_corr
            self.best_epoch = epoch


    def evaluate_from_2way(self, model, y_true_org_dev, y_true_org_test, overall_range, min_score, epoch):
        # Predict Scores
        y_pred_overall_dev, y_pred_anaytic_dev = model.predict(self.dev_features_list)
        y_pred_overall_test, y_pred_anaytic_test = model.predict(self.test_features_list)

        # Rescale overall score
        y_pred_overall_dev = y_pred_overall_dev * (overall_range - 1)
        y_pred_overall_dev = np.round(y_pred_overall_dev)
        y_pred_overall_test = y_pred_overall_test * (overall_range - 1)
        y_pred_overall_test = np.round(y_pred_overall_test)

        # Rescale analytic scores
        y_pred_anaytic_dev = np.sum(y_pred_anaytic_dev * np.arange(0, self.analytic_range), axis=-1)
        y_pred_anaytic_dev = np.round(y_pred_anaytic_dev)
        y_pred_anaytic_test = np.sum(y_pred_anaytic_test * np.arange(0, self.analytic_range), axis=-1)
        y_pred_anaytic_test = np.round(y_pred_anaytic_test)

        # Compiled scores
        y_pred_dev = np.concatenate([y_pred_overall_dev, y_pred_anaytic_dev], axis=1)
        y_pred_test = np.concatenate([y_pred_overall_test, y_pred_anaytic_test], axis=1)

        # Set the minimum score to 0
        y_true_dev = y_true_org_dev - min_score
        y_true_test = y_true_org_test - min_score

        # Calculate metrics
        self.dev_qwk = self.calc_qwk(y_true_dev, y_pred_dev)
        self.dev_lwk = self.calc_lwk(y_true_dev, y_pred_dev)
        self.dev_rmse = self.calc_rmse(y_true_dev, y_pred_dev)
        self.dev_mae = self.calc_mae(y_true_dev, y_pred_dev)
        self.dev_corr = self.calc_corr(y_true_dev, y_pred_dev)

        self.test_qwk = self.calc_qwk(y_true_test, y_pred_test)
        self.test_lwk = self.calc_lwk(y_true_test, y_pred_test)
        self.test_rmse = self.calc_rmse(y_true_test, y_pred_test)
        self.test_mae = self.calc_mae(y_true_test, y_pred_test)
        self.test_corr = self.calc_corr(y_true_test, y_pred_test)

        if np.mean(self.dev_qwk) > self.dev_qwk_mean_best:
            self.dev_qwk_mean_best = np.mean(self.dev_qwk)
            self.best_qwk = self.test_qwk
            self.best_lwk = self.test_lwk
            self.best_rmse = self.test_rmse
            self.best_mae = self.test_mae
            self.best_corr = self.test_corr
            self.best_epoch = epoch+1


    def print_results(self):
        print('Now Best Epoch: {}'.format(self.best_epoch))
        print('DEV_QWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.dev_qwk), np.round(self.dev_qwk, 3)))
        print('DEV_LWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.dev_lwk), np.round(self.dev_lwk, 3)))
        print('DEV_RMSE: mean -> {:.3f}, each item -> {}'.format(np.mean(self.dev_rmse), np.round(self.dev_rmse, 3)))
        print('DEV_MAE:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.dev_mae), np.round(self.dev_mae, 3)))
        print('DEV_CORR: mean -> {:.3f}, each item -> {}'.format(np.mean(self.dev_corr), np.round(self.dev_corr, 3)))
        print('-' * 50)
        print('TEST_QWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.test_qwk), np.round(self.test_qwk, 3)))
        print('TEST_LWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.test_lwk), np.round(self.test_lwk, 3)))
        print('TEST_RMSE: mean -> {:.3f}, each item -> {}'.format(np.mean(self.test_rmse), np.round(self.test_rmse, 3)))
        print('TEST_MAE:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.test_mae), np.round(self.test_mae, 3)))
        print('TEST_CORR: mean -> {:.3f}, each item -> {}'.format(np.mean(self.test_corr), np.round(self.test_corr, 3)))
        print('-' * 50)
        print('BEST_QWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.best_qwk), np.round(self.best_qwk, 3)))
        print('BEST_LWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.best_lwk), np.round(self.best_lwk, 3)))
        print('BEST_RMSE: mean -> {:.3f}, each item -> {}'.format(np.mean(self.best_rmse), np.round(self.best_rmse, 3)))
        print('BEST_MAE:  mean -> {:.3f}, each item -> {}'.format(np.mean(self.best_mae), np.round(self.best_mae, 3)))
        print('BEST_CORR: mean -> {:.3f}, each item -> {}'.format(np.mean(self.best_corr), np.round(self.best_corr, 3)))
        print('-' * 100)
