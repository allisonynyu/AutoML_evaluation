import os
import shutil
import json
import time
from datetime import datetime
import uuid
import pandas as pd
from autonml import AutonML, create_d3m_dataset
from sklearn.metrics import r2_score
from autodiag.utils import retrieve_exam, search, insert_record
import pkg_resources


# the unit of time budget in second (e.g. 600 stands for 600 seconds)
def eval_autonml(openml_id, exam_id, time_budget, n_jobs=8):
    exam_dir = retrieve_exam(openml_id, exam_id)
    train_data_path = exam_dir + '/train.csv'
    test_data_path = exam_dir + '/test.csv'
    d3m_data_path = os.getcwd() + '/d3m_tmp'
    try:
        os.mkdir('d3m_tmp')
    except FileExistsError:
        shutil.rmtree(d3m_data_path)
        os.mkdir('d3m_tmp')
    output_path = os.getcwd() + '/output_tmp'
    try:
        os.mkdir('output_tmp')
    except FileExistsError:
        shutil.rmtree(output_path)
        os.mkdir('output_tmp')
    train = pd.read_csv(train_data_path)
    target = list(train.columns)[-1]
    test = pd.read_csv(test_data_path)
    y_test = test.iloc[:, -1]

    create_d3m_dataset.run(train_data_path, test_data_path, d3m_data_path, target, "rSquared", ["regression"])

    start_time = time.time()  # start
    timeout = time_budget // 60
    reg = AutonML(input_dir=d3m_data_path,
                  output_dir=output_path,
                  timeout=timeout, numcpus=n_jobs)
    reg.run()
    stop_time = time.time()  # stop
    training_time = stop_time - start_time

    best_model, training_score, testing_score = None, None, None
    model_files_path = output_path + '/' + os.listdir(output_path)[0] + '/pipelines_ranked'
    for model_file in os.listdir(model_files_path):
        model = open(model_files_path + '/' + model_file, 'r')
        model_info = json.loads(model.read())
        rank = int(model_info['pipeline_rank'])
        if rank == 1:
            best_model = model_file.split('.')[0]
            training_score = model_info['pipeline_score']
            break
    try:
        testing_predictions = pd.read_csv(output_path + '/' + os.listdir(output_path)[0] + '/predictions/' + best_model
                                          + '.predictions.csv')[target]
        testing_score = r2_score(y_test, testing_predictions)
    except Exception:
        pass

    shutil.rmtree(d3m_data_path)
    shutil.rmtree(output_path)

    automl_name = 'autonml'
    automl_version = pkg_resources.get_distribution("autonml").version
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







