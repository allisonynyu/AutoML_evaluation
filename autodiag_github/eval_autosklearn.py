import time
from datetime import datetime
import uuid
import pandas as pd
import autosklearn
import autosklearn.regression
from autosklearn.metrics import r2
from sklearn.metrics import r2_score
from autodiag.utils import retrieve_exam, search, insert_record

config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'unix_socket': '/Applications/MAMP/tmp/mysql/mysql.sock',
    'database': 'eval_automl',
    'raise_on_warnings': True
}


# the unit of time budget in second (e.g. 600 stands for 600 seconds)
def eval_autosklearn(openml_id, exam_id, time_budget, n_jobs=8):
    exam_dir = retrieve_exam(openml_id, exam_id)
    train = pd.read_csv(exam_dir + '/train.csv')
    test = pd.read_csv(exam_dir + '/test.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    start_time = time.time()  # start
    reg = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=time_budget,
        per_run_time_limit=time_budget/4,
        metric=r2,
    )
    reg.fit(X_train, y_train)
    stop_time = time.time()  # stop
    training_time = stop_time - start_time

    training_predictions = reg.predict(X_train)
    training_score = r2_score(y_train, training_predictions)
    testing_predictions = reg.predict(X_test)
    testing_score = r2_score(y_test, testing_predictions)

    automl_name = 'autosklearn'
    automl_version = autosklearn.__version__
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







