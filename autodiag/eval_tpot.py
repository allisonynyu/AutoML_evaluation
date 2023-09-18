import time
from datetime import datetime
import uuid
import pandas as pd
import tpot
from tpot import TPOTRegressor
from autodiag.utils import retrieve_exam, search, insert_record


# the unit of time budget in second (e.g. 600 stands for 600 seconds)
def eval_tpot(openml_id, exam_id, time_budget, n_jobs=8):
    exam_dir = retrieve_exam(openml_id, exam_id)
    train = pd.read_csv(exam_dir + '/train.csv')
    test = pd.read_csv(exam_dir + '/test.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    print("Training TPOT...")
    start_time = time.time()  # start
    reg = TPOTRegressor(verbosity=0, n_jobs=n_jobs, scoring='r2', random_state=1, max_time_mins=time_budget/60,
                         max_eval_time_mins=0.04, population_size=15)
    reg.fit(X_train, y_train)
    stop_time = time.time()  # stop
    training_time = stop_time - start_time

    training_score = reg.score(X_train, y_train)
    testing_score = reg.score(X_test, y_test)

    automl_name = 'tpot'
    automl_version = tpot.__version__
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







