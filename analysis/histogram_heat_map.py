import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def histogram(time):
    csv_file = 'final_data.csv'
    data = pd.read_csv(csv_file)
            
    autonml, h2o, tpot, ag = [], [], [], []
    competition = data[['aml'+str(time)+'test', 'h2o'+str(time)+'test', 'tpot'+str(time)+'test', 'ag'+str(time)+'test']]
    for _, row in competition.iterrows():
        ranking = rankdata(row, method='min')
        autonml.append(ranking[0])
        h2o.append(ranking[1])
        tpot.append(ranking[2])
        ag.append(ranking[3])

    algo_translation = {
        'xgboost_gbtree': 'XGBoost',
        'ExtraTreesClassifier': 'extra_trees',
        'MLPClassifier': 'mlp',
        'RandomForestClassifier': 'random_forest',
        'RandomForestEntr': 'random_forest',
        'GradientBoostingClassifier': 'gradient_boosting',
        'SGDClassifier': 'sgd',
        'NeuralNetTorch': 'DeepLearning',
        'KNeighborsClassifier': 'KNeighborsDist',
        'LogisticRegression': 'logistic_regression'
    }
    
    autonml_first_algo = data['aml'+str(time)+'algo'][np.array(autonml) == 1].to_numpy()
    (algo_autonml, count_autonml) = np.unique(autonml_first_algo, return_counts=True)
    for algo in algo_autonml:
        if algo in algo_translation.keys():
            index = algo_autonml.tolist().index(algo)
            algo_autonml[index] = algo_translation[algo]
    h2o_first_algo = data['h2o'+str(time)+'algo'][np.array(h2o) == 1].to_numpy()
    (algo_h2o, count_h2o) = np.unique(h2o_first_algo, return_counts=True)
    for algo in algo_h2o:
        if algo in algo_translation.keys():
            index = algo_h2o.tolist().index(algo)
            algo_h2o[index] = algo_translation[algo]
    tpot_first_algo = data['tpot'+str(time)+'algo'][np.array(tpot) == 1].to_numpy()
    (algo_tpot, count_tpot) = np.unique(tpot_first_algo, return_counts=True)
    for algo in algo_tpot:
        if algo in algo_translation.keys():
            index = algo_tpot.tolist().index(algo)
            algo_tpot[index] = algo_translation[algo]
    ag_first_algo = data['ag'+str(time)+'algo'][np.array(ag) == 1].to_numpy()
    (algo_ag, count_ag) = np.unique(ag_first_algo, return_counts=True)
    for algo in algo_ag:
        if algo in algo_translation.keys():
            index = algo_ag.tolist().index(algo)
            algo_ag[index] = algo_translation[algo]
    
    algo_list = ['gradient_boosting', 'extra_trees', 'ada_boost', 'sgd', 'bagging', 'mlp', 'random_forest', 'XGBoost', 'logistic_regression', 'DeepLearning', 'GBM', 'GLM', 'DRF', 'GaussianNB', 'MultinomialNB', 'BernoulliNB', 'XGBClassifier', 'DecisionTreeClassifier', 'WeightedEnsemble_L2', 'LightGBMLarge', 'CatBoost', 'LightGBM', 'KNeighborsDist', 'LightGBMXT']
    count_table = []
    for algo in algo_list:
        algo_counts = []
        try:
            algo_counts.append(count_autonml[algo_autonml.tolist().index(algo)])
        except ValueError:
            algo_counts.append(0)
        try:
            algo_counts.append(count_h2o[algo_h2o.tolist().index(algo)])
        except ValueError:
            algo_counts.append(0)
        try:
            algo_counts.append(count_tpot[algo_tpot.tolist().index(algo)])
        except ValueError:
            algo_counts.append(0)
        try:
            algo_counts.append(count_ag[algo_ag.tolist().index(algo)])
        except ValueError:
            algo_counts.append(0)
        count_table.append(algo_counts)
    
    
    plt.figure(figsize=(20,10))
    df = pd.DataFrame(np.array(count_table).T, columns=algo_list)
    df.insert(0, "AutoML", ['Auto^nML', 'H2O AutoML', 'TPOT', 'AutoGluon'], True)
    df.plot(x='AutoML', kind='barh', stacked=True, title='Frequency of First Place Algorithm ('+str(time)+'s)')
    plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.72), prop={'size': 4.5})
    plt.tight_layout()
    plt.savefig('histogram_'+str(time)+'s.svg')

    for index, row in df.iterrows():
        new_row = row[1:]/df.sum(axis=1)[index]
        df.loc[index] = new_row
    df['AutoML'] = ['Auto^nML', 'H2O AutoML', 'TPOT', 'AutoGluon']
    plt.figure(figsize=(20,10))
    df.plot(x='AutoML', kind='barh', stacked=True, title='Percentage Frequency of First Place Algorithm ('+str(time)+'s)')
    plt.legend(loc='center right', bbox_to_anchor=(1.2, 0.72), prop={'size': 4.5})
    plt.tight_layout()
    plt.savefig('histogram_percentage_'+str(time)+'s.svg')

    autonml_win = np.array(autonml) == 1
    h2o_win = np.array(h2o) == 1
    tpot_win = np.array(tpot) == 1
    ag_win = np.array(ag) == 1
    algo_count = np.zeros((len(algo_list), len(algo_list)))
    for i in range(autonml_win.shape[0]):
        if not autonml_win[i]:
            autonml_algo = data.iloc[i]['aml'+str(time)+'algo']
            if autonml_algo not in algo_list:
                autonml_algo = algo_translation[autonml_algo]
            autonml_algo_index = algo_list.index(autonml_algo)

            if h2o_win[i]:
                h2o_algo = data.iloc[i]['h2o'+str(time)+'algo']
                if h2o_algo not in algo_list:
                    h2o_algo = algo_translation[h2o_algo]
                h2o_algo_index = algo_list.index(h2o_algo)
                algo_count[autonml_algo_index, h2o_algo_index] += 1
            if tpot_win[i]:
                tpot_algo = data.iloc[i]['tpot'+str(time)+'algo']
                if tpot_algo not in algo_list:
                    tpot_algo = algo_translation[tpot_algo]
                tpot_algo_index = algo_list.index(tpot_algo)
                algo_count[autonml_algo_index, tpot_algo_index] += 1
            if ag_win[i]:
                ag_algo = data.iloc[i]['ag'+str(time)+'algo']
                if ag_algo not in algo_list:
                    ag_algo = algo_translation[ag_algo]
                ag_algo_index = algo_list.index(ag_algo)
                algo_count[autonml_algo_index, ag_algo_index] += 1
    
    seaborn_data = []
    for i in range(algo_count.shape[0]):
        for j in range(algo_count[0].shape[0]):
            seaborn_data.append([algo_list[i], algo_list[j], algo_count[i][j]])
    seaborn_data = pd.DataFrame(seaborn_data, columns=['Auto^nML Core Algorithm', 'Top AutoML Core Algorithm', 'Count'])
    seaborn_data['Count'] = seaborn_data['Count'].astype(int)
    plt.figure(figsize=(10,10))
    seaborn_data = seaborn_data.pivot('Top AutoML Core Algorithm', 'Auto^nML Core Algorithm', 'Count')
    ax = sns.heatmap(seaborn_data, annot=True, fmt="d", cmap="YlGnBu")
    ax.set(title='Auto^nML Core Algorithm V.S. Top AutoML Core Algorithm when Auto^nML does not Get the First Place ('+str(time)+'s)')
    plt.tight_layout()
    plt.savefig('heatmap_'+str(time)+'s.svg')

if __name__ == "__main__":
    times = [60, 600, 1200]
    for time in times:
        histogram(time)