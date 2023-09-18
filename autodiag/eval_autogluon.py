import os
import shutil
import time
from datetime import datetime
import uuid
import pandas as pd
from autogluon.tabular import TabularPredictor
from autodiag.utils import retrieve_exam, search, insert_record
import pkg_resources

config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'unix_socket': '/Applications/MAMP/tmp/mysql/mysql.sock',
    'database': 'eval_automl',
    'raise_on_warnings': True
}


# the unit of time budget in second (e.g. 600 stands for 600 seconds)
def eval_autogluon(openml_id, exam_id, time_budget, n_jobs=8):
    exam_dir = retrieve_exam(openml_id, exam_id)
    train = pd.read_csv(exam_dir + '/train.csv')
    test = pd.read_csv(exam_dir + '/test.csv')

    save_path = os.getcwd() + '/save_tmp'
    try:
        os.mkdir('save_tmp')
    except FileExistsError:
        shutil.rmtree(save_path)
        os.mkdir('save_tmp')

    start_time = time.time()  # start
    label = train.columns[-1]
    metric = 'r2'
    reg = TabularPredictor(label, eval_metric=metric, path=save_path).fit(train, time_limit=time_budget,
                                                                ag_args_fit={'num_cpus': n_jobs})
    stop_time = time.time()  # stop
    training_time = stop_time - start_time

    train_leaderboard = reg.leaderboard()
    training_score = train_leaderboard.score_val[0]
    top_model = train_leaderboard.model[0]
    test_leaderboard = reg.leaderboard(test)
    testing_score = test_leaderboard.score_test.loc[test_leaderboard['model'] == top_model]

    shutil.rmtree(save_path)

    automl_name = 'autogluon'
    automl_version = pkg_resources.get_distribution("autogluon").version
    pipeline_building = None
    testing_datetime = datetime.now()

    search_query = "SELECT experiment_id FROM evaluation_results"
    existing_experiment_records = search(search_query)
    existing_experiment_ids = [experiment_id[0] for experiment_id in existing_experiment_records]
    experiment_id = str(uuid.uuid4())
    while experiment_id in existing_experiment_ids:
        experiment_id = str(uuid.uuid4())

    record = {
        "experiment_id": experiment_id,
        "openml_id": openml_id,
        "exam_id": exam_id,
        "time_budget": time_budget,
        "automl_name": automl_name,
        "automl_version": automl_version,
        "training_time": round(float(training_time), 2),
        "training_score": round(float(training_score), 4),
        "testing_score": round(float(testing_score), 4),
        "pipeline_building": pipeline_building,
        "testing_datetime": testing_datetime,
    }
    table_name = 'evaluation_results'
    insert_record(record, table_name)







