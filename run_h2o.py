import os
from src.utils import add_exam
#from src.eval_tpot import eval_tpot
#from src.eval_autogluon import eval_autogluon
from src.eval_h2o import eval_h2o

time_budgets = [60, 600, 1200]
eval_list = [8,189,190,191,192,193,194,195,196,197,198,199,
             200,201,203,204,206,207,208,209,211,212,213,214,215,216,217,218,
             222,223,225,226,227,228,229,230,231,232,294,299,344,420,422,482,491,492,494,497,
             500,506,507,509,511,513,518,520,521,522,523,524,526,527,528,
             531,533,534,535,536,541,543,546,547,549,551,553,555,556,557,560,561,562,566,567,568]
test_size = 0.25

#add_exam(eval_list,os.getcwd()+'/tests', [test_size])
for time_budget in time_budgets:
    for openml_id in eval_list:
        try:
            exam_id = os.listdir(os.getcwd()+'/tests/'+str(openml_id))[-1]
            eval_h2o(openml_id, exam_id, time_budget)
        except Exception:
            print("Error: ", openml_id, time_budget)