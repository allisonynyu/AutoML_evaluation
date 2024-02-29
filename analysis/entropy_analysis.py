import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def plot():
    data = pd.read_csv("final_data.csv")
    aml60, aml600, aml1200 = data['aml60test'], data['aml600test'], data['aml1200test']
    h2o60, h2o600, h2o1200 = data['h2o60test'], data['h2o600test'], data['h2o1200test']
    diff60, diff600, diff1200 = aml60 - h2o60, aml600 - h2o600, aml1200 - h2o1200
    
    amlalgo = data[['aml60algo','aml600algo', 'aml1200algo']]
    h2oalgo = data[['h2o60algo','h2o600algo', 'h2o1200algo']]
    aml_entropy, h2o_entropy = [], []
    for row in amlalgo.values.tolist():
        entr = entropy1(row, base=3)
        aml_entropy.append(entr)
    for row in h2oalgo.values.tolist():
        entr = entropy1(row, base=3)
        h2o_entropy.append(entr)

    plt.figure()
    plt.scatter(diff60, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.title('AutonML Entropy - AutonML and H2O difference 60s')
    plt.savefig('AutonML Entropy - AutonML and H2O difference 60s.svg')
    plt.figure()
    plt.scatter(diff60, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.title('H2O Entropy - AutonML and H2O difference 60s')
    plt.savefig('H2O Entropy - AutonML and H2O difference 60s.svg')
    
    plt.figure()
    plt.scatter(diff600, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.title('AutonML Entropy - AutonML and H2O difference 600s')
    plt.savefig('AutonML Entropy - AutonML and H2O difference 600s.svg')
    plt.figure()
    plt.scatter(diff60, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.title('H2O Entropy - AutonML and H2O difference 600s.svg')
    plt.savefig('H2O Entropy - AutonML and H2O difference 600s.svg')
    
    plt.figure()
    plt.scatter(diff1200, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.title('AutonML Entropy - AutonML and H2O difference 1200s')
    plt.savefig('AutonML Entropy - AutonML and H2O difference 1200s.svg')
    plt.figure()
    plt.scatter(diff1200, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.title('H2O Entropy - AutonML and H2O difference 1200s')
    plt.savefig('H2O Entropy - AutonML and H2O difference 1200s.svg')

    plt.figure()
    plt.scatter(diff60, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.scatter(diff60, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.xlim([-0.12, 0.12])
    plt.title('Partial Entropy - AutonML and H2O difference 60s.svg')
    plt.savefig('Partial Entropy-difference 60s.svg')
    
    plt.figure()
    plt.scatter(diff600, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.scatter(diff60, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.xlim([-0.12, 0.12])
    plt.title('Partial Entropy - AutonML and H2O difference 600s.svg')
    plt.savefig('Partial Entropy-difference 600s.svg')
    
    plt.figure()
    plt.scatter(diff1200, aml_entropy, color='r', label='Auto^nML Entropy', alpha=.3)
    plt.scatter(diff1200, h2o_entropy, color='b', label='H2O Entropy', alpha=.3)
    plt.xlim([-0.12, 0.12])
    plt.title('Partial Entropy - AutonML and H2O difference 1200s.svg')
    plt.savefig('Partial Entropy-difference 1200s.svg')


if __name__ == "__main__":
    plot()
