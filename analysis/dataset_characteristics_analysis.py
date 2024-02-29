import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import glob
import copy
from datetime import datetime

dim_interval = 10
dim_split = list(range(10, 125, dim_interval))
inst_interval = 1000
inst_split = list(range(1000, 10000, inst_interval))

def plot_graph(autonml_avg, autonml_median, autonml_ci,
               h2o_avg, h2o_median, h2o_ci,
               tpot_avg, tpot_median, tpot_ci,
               ag_avg, ag_median, ag_ci,
               inputx, mode, csv_file, timestamp):
        
        plt.figure(figsize=(6.5, 6.5))
        
        ax = plt.gca()
        
        plt.errorbar(inputx, autonml_avg, yerr=autonml_ci, color='r', label='Auto^nML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(inputx + 0.05 * (inputx[-1] - inputx[-2]), h2o_avg, yerr=h2o_ci, color='b', label='H2O AutoML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(inputx + 0.1 * (inputx[-1] - inputx[-2]), tpot_avg, yerr=tpot_ci, color='g', label='TPOT mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(inputx + 0.15 * (inputx[-1] - inputx[-2]), ag_avg, yerr=ag_ci, color='m', label='AutoGluon mean', alpha=.5, fmt='o', markersize=8, capsize=10)

        plt.xlim([0, inputx[-1] + 5 * (inputx[-1] - inputx[-2])])
        plt.ylim([3.9, 0.8])
        plt.xlabel("Training Time Limit (seconds)")
        if mode == "dimensionality":
            plt.xlabel("Dimensionality of datasets", fontsize=14)
        elif mode == "number of intances":
            plt.xlabel("Number of Instances of datasets", fontsize=14)
        plt.ylabel("Rank from Test Predictions accross datasets", fontsize=14)
        #plt.title("Average and Median Rank per AutoML on Test Predictions (" + str(timelimit) + "s)")
        plt.title("Average rank per AutoML on Test Predictions", fontsize=16)
        plt.legend(['Auto^nML', 'H2O AutoML', 'TPOT', 'AutoGluon'])
        ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.5))

        plt.savefig(mode + "-" + "max" +str(inputx[-1]) + "-" + "interval" +str(inputx[-1] - inputx[-2]) + "-" + csv_file + timestamp + ".svg")

def bin_index(num, break_points):
    index = 0
    for break_point in break_points:
        if num >= break_point:
            index += 1
        else:
            break
    return index

def plot():

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H:%M:%S")

    csv_files = glob.glob('*.csv')
    for csv_file in csv_files:
        if csv_file == "final_data":
            
            data = pd.read_csv(csv_file)

            dim_bin, inst_bin = [], []
            for _ in range(len(dim_split)+1):
                dim_bin.append([])
            for _ in range(len(inst_split)+1):
                inst_bin.append([])
            
            for index, row in data.iterrows():
                number_of_features = row['NumberOfFeatures']
                dim_index = bin_index(number_of_features, dim_split)
                dim_bin[dim_index].append(index)

                number_of_instances = row['NumberOfInstances']
                inst_index = bin_index(number_of_instances, inst_split)
                inst_bin[inst_index].append(index)
            
            autonml_60s, h2o_60s, tpot_60s, ag_60s = [], [], [], []
            competition_60s = data[['aml60test', 'h2o60test', 'tpot60test', 'ag60test']]
            for _, row in competition_60s.iterrows():
                ranking = rankdata(row)
                autonml_60s.append(ranking[0])
                h2o_60s.append(ranking[1])
                tpot_60s.append(ranking[2])
                ag_60s.append(ranking[3])
            
            autonml_600s, h2o_600s, tpot_600s, ag_600s = [], [], [], []
            competition_600s = data[['aml600test', 'h2o600test', 'tpot600test', 'ag600test']]
            for _, row in competition_600s.iterrows():
                ranking = rankdata(row)
                autonml_600s.append(ranking[0])
                h2o_600s.append(ranking[1])
                tpot_600s.append(ranking[2])
                ag_600s.append(ranking[3])
            
            autonml_1200s, h2o_1200s, tpot_1200s, ag_1200s = [], [], [], []
            competition_1200s = data[['aml1200test', 'h2o1200test', 'tpot1200test', 'ag1200test']]
            for _, row in competition_1200s.iterrows():
                ranking = rankdata(row)
                autonml_1200s.append(ranking[0])
                h2o_1200s.append(ranking[1])
                tpot_1200s.append(ranking[2])
                ag_1200s.append(ranking[3])

            autonml_avg = np.mean(np.array([autonml_60s, autonml_600s, autonml_1200s]), axis=0)
            h2o_avg = np.mean(np.array([h2o_60s, h2o_600s, h2o_1200s]), axis=0)
            tpot_avg = np.mean(np.array([tpot_60s, tpot_600s, tpot_1200s]), axis=0)
            ag_avg = np.mean(np.array([ag_60s, ag_600s, ag_1200s]), axis=0)

            autonml_avg_dim, autonml_median_dim, autonml_ci_dim = [], [], []
            h2o_avg_dim, h2o_median_dim, h2o_ci_dim = [], [], []
            tpot_avg_dim, tpot_median_dim, tpot_ci_dim = [], [], []
            ag_avg_dim, ag_median_dim, ag_ci_dim = [], [], []
            
            z = 1.96 # 95% confidence

            for b in dim_bin:
                if b != []:
                    autonml_bin = np.array(autonml_avg)[np.array(b)]
                    h2o_bin = np.array(h2o_avg)[np.array(b)]
                    tpot_bin = np.array(tpot_avg)[np.array(b)]
                    ag_bin = np.array(ag_avg)[np.array(b)]
                    length = len(autonml_avg)
                    autonml_avg_dim.append(np.mean(autonml_bin))
                    autonml_median_dim.append(np.median(autonml_bin))
                    autonml_ci_dim.append(z * np.std(autonml_bin) / np.sqrt(length))
                    h2o_avg_dim.append(np.mean(h2o_bin))
                    h2o_median_dim.append(np.median(h2o_bin))
                    h2o_ci_dim.append(z * np.std(h2o_bin) / np.sqrt(length))
                    tpot_avg_dim.append(np.mean(tpot_bin))
                    tpot_median_dim.append(np.median(tpot_bin))
                    tpot_ci_dim.append(z * np.std(tpot_bin) / np.sqrt(length))
                    ag_avg_dim.append(np.mean(ag_bin))
                    ag_median_dim.append(np.median(ag_bin))
                    ag_ci_dim.append(z * np.std(ag_bin) / np.sqrt(length))
                else:
                    autonml_avg_dim.append(-1)
                    autonml_median_dim.append(-1)
                    autonml_ci_dim.append(0)
                    h2o_avg_dim.append(-1)
                    h2o_median_dim.append(-1)
                    h2o_ci_dim.append(0)
                    tpot_avg_dim.append(-1)
                    tpot_median_dim.append(-1)
                    tpot_ci_dim.append(0)
                    ag_avg_dim.append(-1)
                    ag_median_dim.append(-1)
                    ag_ci_dim.append(0)

            dimensionality = copy.deepcopy(dim_split)
            dimensionality.insert(0, 0)
            dimensionality = np.array(dimensionality) + int(dim_interval/2)
            
            plot_graph(autonml_avg_dim, autonml_median_dim, autonml_ci_dim, h2o_avg_dim, h2o_median_dim, h2o_ci_dim, tpot_avg_dim, tpot_median_dim, tpot_ci_dim, ag_avg_dim, ag_median_dim, ag_ci_dim, dimensionality, "dimensionality", csv_file, timestamp)

            
            autonml_avg_inst, autonml_median_inst, autonml_ci_inst = [], [], []
            h2o_avg_inst, h2o_median_inst, h2o_ci_inst = [], [], []
            tpot_avg_inst, tpot_median_inst, tpot_ci_inst = [], [], []
            ag_avg_inst, ag_median_inst, ag_ci_inst = [], [], []
            
            z = 1.96 # 95% confidence

            for b in inst_bin:
                if b != []:
                    autonml_bin = np.array(autonml_avg)[np.array(b)]
                    h2o_bin = np.array(h2o_avg)[np.array(b)]
                    tpot_bin = np.array(tpot_avg)[np.array(b)]
                    ag_bin = np.array(ag_avg)[np.array(b)]
                    length = len(autonml_avg)
                    autonml_avg_inst.append(np.mean(autonml_bin))
                    autonml_median_inst.append(np.median(autonml_bin))
                    autonml_ci_inst.append(z * np.std(autonml_bin) / np.sqrt(length))
                    h2o_avg_inst.append(np.mean(h2o_bin))
                    h2o_median_inst.append(np.median(h2o_bin))
                    h2o_ci_inst.append(z * np.std(h2o_bin) / np.sqrt(length))
                    tpot_avg_inst.append(np.mean(tpot_bin))
                    tpot_median_inst.append(np.median(tpot_bin))
                    tpot_ci_inst.append(z * np.std(tpot_bin) / np.sqrt(length))
                    ag_avg_inst.append(np.mean(ag_bin))
                    ag_median_inst.append(np.median(ag_bin))
                    ag_ci_inst.append(z * np.std(ag_bin) / np.sqrt(length))
                else:
                    autonml_avg_inst.append(-1)
                    autonml_median_inst.append(-1)
                    autonml_ci_inst.append(0)
                    h2o_avg_inst.append(-1)
                    h2o_median_inst.append(-1)
                    h2o_ci_inst.append(0)
                    tpot_avg_inst.append(-1)
                    tpot_median_inst.append(-1)
                    tpot_ci_inst.append(0)
                    ag_avg_inst.append(-1)
                    ag_median_inst.append(-1)
                    ag_ci_inst.append(0)

            numberOfIntances = copy.deepcopy(inst_split)
            numberOfIntances.insert(0, 0)
            numberOfIntances = np.array(numberOfIntances) + int(inst_interval/2)
                
            plot_graph(autonml_avg_inst, autonml_median_inst, autonml_ci_inst, h2o_avg_inst, h2o_median_inst, h2o_ci_inst, tpot_avg_inst, tpot_median_inst, tpot_ci_inst, ag_avg_inst, ag_median_inst, ag_ci_inst, numberOfIntances, "number of intances", csv_file, timestamp)

if __name__ == "__main__":
    plot()