import time
from datetime import datetime
import uuid
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import r2_score
from src.utils import retrieve_exam, search, insert_record


# the unit of time budget in second (e.g. 600 stands for 600 seconds)
def eval_h2o(openml_id, exam_id, time_budget, n_jobs=8):
    exam_dir = retrieve_exam(openml_id, exam_id)
    train = pd.read_csv(exam_dir + '/train.csv')
    test = pd.read_csv(exam_dir + '/test.csv')
    y_train = train.iloc[:, -1]
    y_test = test.iloc[:, -1]

    h2o.init()
    h2o_train = h2o.import_file(path=exam_dir + '/train.csv')
    target = h2o_train.names[-1]
    h2o_test = h2o.import_file(path=exam_dir + '/test.csv')

    # train AutoML
    start_time = time.time()  # start
    reg = H2OAutoML(max_runtime_secs=time_budget, nfolds=0, sort_metric='R2')
    reg.train(y=target, training_frame=h2o_train)
    stop_time = time.time()  # stop
    training_time = stop_time - start_time

    best_model = reg.leader
    training_predictions = best_model.predict(h2o_train).as_data_frame(use_pandas=True)
    training_score = r2_score(y_train, training_predictions)
    testing_predictions = best_model.predict(h2o_test).as_data_frame(use_pandas=True)
    testing_score = r2_score(y_test, testing_predictions)
    pipeline_building = best_model.model_id

    automl_name = 'h2o'
    automl_version = h2o.__version__
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







