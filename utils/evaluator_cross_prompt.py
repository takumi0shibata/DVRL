from utils.general_utils import rescale_single_attribute
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.general_utils import get_min_max_scores_for_rubric
import numpy as np
import os
import pandas as pd


class Evaluator():

    def __init__(self, test_prompt_id, X_dev_prompt_ids, X_test_prompt_ids, dev_features_list, test_features_list,
                 Y_dev, Y_test, attribute_name, seed, essay_ids_dev, essay_ids_test, use_rubric):
        self.attribute_name = attribute_name
        self.seed = seed
        self.test_prompt_id = test_prompt_id
        self.dev_features_list = dev_features_list
        self.test_features_list = test_features_list
        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.essay_ids_dev, self.essay_ids_test = essay_ids_dev, essay_ids_test
        self.use_rubric = use_rubric
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_test_flat = Y_test.flatten()
        self.Y_test_org = rescale_single_attribute(self.Y_test_flat, self.X_test_prompt_ids, attribute_name, rubric=self.use_rubric)
        self.Y_dev_flat = Y_dev.flatten()
        self.Y_dev_org = rescale_single_attribute(self.Y_dev_flat, self.X_dev_prompt_ids, attribute_name, rubric=self.use_rubric)

        self.best_dev_qwk = -1
        self.best_test_qwk = -1
        self.best_epoch = -1

    def calc_qwk(self, y_true, y_pred, prompt_ids):
        qwk_list = []
        for prompt_id in range(1, 9):
            min_score, max_score = get_min_max_scores_for_rubric()[prompt_id][self.attribute_name]
            if np.sum(prompt_ids==prompt_id) == 0:
                continue
            kappa_score = cohen_kappa_score(y_true[prompt_ids==prompt_id], y_pred[prompt_ids==prompt_id], weights='quadratic', labels=[i for i in range(min_score, max_score+1)])
            qwk_list.append(kappa_score)

        print(qwk_list)
        return np.mean(np.array(qwk_list))

    def calc_lwk(self, y_true, y_pred, prompt_ids):
        lwk_list = []
        for prompt_id in range(1, 9):
            min_score, max_score = get_min_max_scores_for_rubric()[prompt_id][self.attribute_name]
            if np.sum(prompt_ids==prompt_id) == 0:
                continue
            kappa_score = cohen_kappa_score(y_true[prompt_ids==prompt_id], y_pred[prompt_ids==prompt_id], weights='linear', labels=[i for i in range(min_score, max_score+1)])
            lwk_list.append(kappa_score)

        return np.mean(np.array(lwk_list))
    
    def calc_corr(self, y_true, y_pred):
        corr_score = np.corrcoef(y_true, y_pred)[0, 1]

        return corr_score

    def calc_rmse(self, y_true, y_pred):
        rmse_score = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return rmse_score

    def calc_mae(self, y_true, y_pred):
        mae_score = mean_absolute_error(y_true, y_pred)

        return mae_score

    def evaluate(self, model, epoch):
        dev_pred = model.predict(self.dev_features_list)
        test_pred = model.predict(self.test_features_list)

        test_pred_flat = test_pred.flatten()
        test_pred_org = rescale_single_attribute(test_pred_flat, self.X_test_prompt_ids, self.attribute_name, rubric=self.use_rubric)
        dev_pred_flat = dev_pred.flatten()
        dev_pred_org = rescale_single_attribute(dev_pred_flat, self.X_dev_prompt_ids, self.attribute_name, rubric=self.use_rubric)
        print(test_pred_org)

        self.dev_qwk = self.calc_qwk(self.Y_dev_org, dev_pred_org, np.array(self.X_dev_prompt_ids))
        self.test_qwk = self.calc_qwk(self.Y_test_org, test_pred_org, np.array(self.X_test_prompt_ids))
        self.dev_lwk = self.calc_lwk(self.Y_dev_org, dev_pred_org, np.array(self.X_dev_prompt_ids))
        self.test_lwk = self.calc_lwk(self.Y_test_org, test_pred_org, np.array(self.X_test_prompt_ids))
        self.dev_corr = self.calc_corr(self.Y_dev_flat, dev_pred_flat)
        self.test_corr = self.calc_corr(self.Y_test_flat, test_pred_flat)
        self.dev_rmse = self.calc_rmse(self.Y_dev_flat, dev_pred_flat)
        self.test_rmse = self.calc_rmse(self.Y_test_flat, test_pred_flat)
        self.dev_mae = self.calc_mae(self.Y_dev_flat, dev_pred_flat)
        self.test_mae = self.calc_mae(self.Y_test_flat, test_pred_flat)

        if self.dev_qwk > self.best_dev_qwk:
            self.best_dev_qwk = self.dev_qwk
            self.best_test_qwk = self.test_qwk
            self.best_dev_lwk = self.dev_lwk
            self.best_test_lwk = self.test_lwk
            self.best_dev_corr = self.dev_corr
            self.best_test_corr = self.test_corr
            self.best_dev_rmse = self.dev_rmse
            self.best_test_rmse = self.test_rmse
            self.best_dev_mae = self.dev_mae
            self.best_test_mae = self.test_mae
            self.best_epoch = epoch + 1
        
        self.print_info()

    def print_info(self):
        print('Prompt: {}, Attribute: {}, Seed: {}'.format(self.test_prompt_id, self.attribute_name, self.seed))
        print('DEV QWK: {}'.format(self.dev_qwk))
        print('DEV LWK: {}'.format(self.dev_lwk))
        print('DEV CORR: {}'.format(self.dev_corr))
        print('DEV RMSE: {}'.format(self.dev_rmse))
        print('DEV MAE: {}'.format(self.dev_mae))
        print('-'*50)
        print('TEST QWK: {}'.format(self.test_qwk))
        print('TEST LWK: {}'.format(self.test_lwk))
        print('TEST CORR: {}'.format(self.test_corr))
        print('TEST RMSE: {}'.format(self.test_rmse))
        print('TEST MAE: {}'.format(self.test_mae))
        print('-'*50)
        print('BEST QWK: {}'.format(self.best_test_qwk))
        print('BEST LWK: {}'.format(self.best_test_lwk))
        print('BEST CORR: {}'.format(self.best_test_corr))
        print('BEST RMSE: {}'.format(self.best_test_rmse))
        print('BEST MAE: {}'.format(self.best_test_mae))
        print('-'*100)

    def print_final_info(self):
        print('-'*100)
        print('Prompt: {}, Attribute: {}'.format(self.test_prompt_id, self.attribute_name))
        print('Best Epoch {}:'.format(self.best_epoch))
        print('BEST DEV QWK: {}'.format(self.best_dev_qwk))
        print('BEST DEV LWK: {}'.format(self.best_dev_lwk))
        print('BEST DEV CORR: {}'.format(self.best_dev_corr))
        print('BEST DEV RMSE: {}'.format(self.best_dev_rmse))
        print('BEST DEV MAE: {}'.format(self.best_dev_mae))
        print('-'*50)
        print('BEST TEST QWK: {}'.format(self.best_test_qwk))
        print('BEST TEST LWK: {}'.format(self.best_test_lwk))
        print('BEST TEST CORR: {}'.format(self.best_test_corr))
        print('BEST TEST RMSE: {}'.format(self.best_test_rmse))
        print('BEST TEST MAE: {}'.format(self.best_test_mae))

    def get_best_result(self):
        dev_qwk = self.best_dev_qwk
        dev_lwk = self.best_dev_lwk
        dev_corr = self.best_dev_corr
        dev_rmse = self.best_dev_rmse
        dev_mae = self.best_dev_mae

        test_qwk = self.best_test_qwk
        test_lwk = self.best_test_lwk
        test_corr = self.best_test_corr
        test_rmse = self.best_test_rmse
        test_mae = self.best_test_mae

        best_dev = np.array([dev_qwk, dev_lwk, dev_corr, dev_rmse, dev_mae]).reshape(1, 5)
        best_test = np.array([test_qwk, test_lwk, test_corr, test_rmse, test_mae]).reshape(1, 5)

        return best_dev, best_test