import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import rankdata, sem
import glob

def train_test_auc():
    #os.chdir(split_folder)
    csv_files = glob.glob('*.csv')
    for csv_file in csv_files:
        if csv_file[4:10] != "concat":
            continue
        data = pd.read_csv(csv_file)
        autonml_train_auc = (data['aml60train'] + data['aml600train'] + data['aml1200train'])/3
        autonml_test_auc = (data['aml60test'] + data['aml600test'] + data['aml1200test'])/3
        h2o_train_auc = (data['h2o60train'] + data['h2o600train'] + data['h2o1200train'])/3
        h2o_test_auc = (data['h2o60test'] + data['h2o600test'] + data['h2o1200test'])/3
        tpot_train_auc = (data['tpot60train'] + data['tpot600train'] + data['tpot1200train'])/3
        tpot_test_auc = (data['tpot60test'] + data['tpot600test'] + data['tpot1200test'])/3
        ag_train_auc = (data['ag60train'] + data['ag600train'] + data['ag1200train'])/3
        ag_test_auc = (data['ag60test'] + data['ag600test'] + data['ag1200test'])/3

        reg = LinearRegression()

        X_autonml = np.array(autonml_train_auc).reshape(-1,1)
        y_autonml = np.array(autonml_test_auc)
        reg.fit(X_autonml, y_autonml)
        r2_autonml = reg.score(X_autonml, y_autonml)

        X_h2o = np.array(h2o_train_auc).reshape(-1,1)
        y_h2o = np.array(h2o_test_auc)
        reg.fit(X_h2o, y_h2o)
        r2_h2o = reg.score(X_h2o, y_h2o)

        X_tpot = np.array(tpot_train_auc).reshape(-1,1)
        y_tpot = np.array(tpot_test_auc)
        reg.fit(X_tpot, y_tpot)
        r2_tpot = reg.score(X_tpot, y_tpot)

        X_ag = np.array(ag_train_auc).reshape(-1,1)
        y_ag = np.array(ag_test_auc)
        reg.fit(X_ag, y_ag)
        r2_ag = reg.score(X_ag, y_ag)

        plt.figure(figsize=(6, 6))
        lw = 2
        plt.scatter(autonml_train_auc, autonml_test_auc, color='r', label='Auto^nML (r2_score = %0.2f)' % r2_autonml, alpha=.5)
        plt.scatter(h2o_train_auc, h2o_test_auc, color='b', label='H2O AutoML (r2_score = %0.2f)' % r2_h2o, alpha=.5)
        plt.scatter(tpot_train_auc, tpot_test_auc, color='g', label='TPOT(r2_score = %0.2f)' % r2_tpot, alpha=.5)
        plt.scatter(ag_train_auc, ag_test_auc, color='m', label='AutoGluon (r2_score = %0.2f)' % r2_ag, alpha=.5)
        plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.3, 1.1])
        plt.ylim([0.3, 1.1])
        ax = plt.gca()
        ax.set_aspect(1)
        plt.xlabel("Avg. AUC from training predictions per Dataset", fontsize=14)
        plt.ylabel("Avg. AUC from testing predictions per Dataset", fontsize=14)
        plt.title("Average AUC per AutoML on Test Predictions versus Train Predictions", fontsize=11)
        plt.legend(loc="upper left", prop={'size': 9})
        plt.savefig("train_test_auc " + csv_file[:-4] + '.svg')

def rank():
    csv_files = glob.glob('*.csv')
    for csv_file in csv_files:
        if csv_file[4:10] != "concat":
            continue
        
        data = pd.read_csv(csv_file)
        
        time_budget_autonml = np.array([60, 600, 1200])
        time_budget_h2o= np.array([100, 640, 1240])
        time_budget_tpot = np.array([140, 680, 1280])
        time_budget_ag = np.array([180, 720, 1320])

        competition_60s = data[['aml60test', 'h2o60test', 'tpot60test', 'ag60test']]
        autonml_60s, h2o_60s, tpot_60s, ag_60s = [], [], [], []
        for index, row in competition_60s.iterrows():
            ranking = rankdata(row)
            autonml_60s.append(ranking[0])
            h2o_60s.append(ranking[1])
            tpot_60s.append(ranking[2])
            ag_60s.append(ranking[3])

        competition_600s = data[['aml600test', 'h2o600test', 'tpot600test', 'ag600test']]
        autonml_600s, h2o_600s, tpot_600s, ag_600s = [], [], [], []
        for index, row in competition_600s.iterrows():
            ranking = rankdata(row)
            autonml_600s.append(ranking[0])
            h2o_600s.append(ranking[1])
            tpot_600s.append(ranking[2])
            ag_600s.append(ranking[3])

        competition_1200s = data[['aml1200test', 'h2o1200test', 'tpot1200test', 'ag1200test']]
        autonml_1200s, h2o_1200s, tpot_1200s, ag_1200s = [], [], [], []
        for index, row in competition_1200s.iterrows():
            ranking = rankdata(row)
            autonml_1200s.append(ranking[0])
            h2o_1200s.append(ranking[1])
            tpot_1200s.append(ranking[2])
            ag_1200s.append(ranking[3])

        z = 1.96 # 95% confidence
        length = len(autonml_60s)
        autonml_ranking = np.array([autonml_60s, autonml_600s, autonml_1200s])
        autonml_avg = np.mean(autonml_ranking, axis=1)
        autonml_median = np.median(autonml_ranking, axis=1)
        autonml_ci = z * np.std(autonml_ranking, axis=1) / np.sqrt(length)

        h2o_ranking = np.array([h2o_60s, h2o_600s, h2o_1200s])
        h2o_avg = np.mean(h2o_ranking, axis=1)
        h2o_median = np.median(h2o_ranking, axis=1)
        h2o_ci = z * np.std(h2o_ranking, axis=1) / np.sqrt(length)

        tpot_ranking = np.array([tpot_60s, tpot_600s, tpot_1200s])
        tpot_avg = np.mean(tpot_ranking, axis=1)
        tpot_median = np.median(tpot_ranking, axis=1)
        tpot_ci = z * np.std(tpot_ranking, axis=1) / np.sqrt(length)

        ag_ranking = np.array([ag_60s, ag_600s, ag_1200s])
        ag_avg = np.mean(ag_ranking, axis=1)
        ag_median = np.median(ag_ranking, axis=1)
        ag_ci = z * np.std(ag_ranking, axis=1) / np.sqrt(length)

        plt.figure(figsize=(6, 6))
        
        ax = plt.gca()

        plt.errorbar(time_budget_autonml, autonml_avg, yerr=autonml_ci, color='r', label='Auto^nML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(time_budget_h2o, h2o_avg, yerr=h2o_ci, color='b', label='H2O AutoML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(time_budget_tpot, tpot_avg, yerr=tpot_ci, color='g', label='TPOT mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(time_budget_ag, ag_avg, yerr=ag_ci, color='m', label='AutoGluon mean', alpha=.5, fmt='o', markersize=8, capsize=10)

        plt.xlim([0, 1400])
        plt.ylim([3.2, 0.8])
        plt.xlabel("Training Time Limit (seconds)", fontsize=14)
        plt.ylabel("Rank from Test Predictions accross datasets", fontsize=14)
        plt.title("Average rank per AutoML on Test Predictions", fontsize=14)
        plt.legend(['Auto^nML', 'H2O AutoML', 'TPOT', 'AutoGluon'])
        ax.legend(loc='upper right')

        plt.savefig("ranking " + csv_file[:-4] + '.svg')

if __name__ == "__main__":
    train_test_auc()
    rank()