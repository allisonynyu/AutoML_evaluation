import os
import shutil
import pandas as pd
import openml as oml
import uuid
import mysql.connector
from sklearn.model_selection import train_test_split

# Database schema:

# Table "exams_info":
# openml_id (primary key), exam_id (primary key), test_size, file_path

# Table "evaluation_results":
# experiment_id (primary key), openml_id, exam_id, time_budget, automl_name, automl_version, training_time,
# training_score, testing_score, pipeline_building, testing_datetime

# Replace with your dataset configuration here
config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'unix_socket': '/Applications/MAMP/tmp/mysql/mysql.sock',
    'database': 'eval_automl',
    'raise_on_warnings': True
}


def add_exam(openml_id_list, output_dir, test_size_list, num_of_tests=1, database_config=config):
    for openml_id in openml_id_list:
        dataset = oml.datasets.get_dataset(openml_id)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        dataset_dir = output_dir + '/' + str(openml_id)
        if not os.path.isdir(dataset_dir):
            os.mkdir(dataset_dir)
        for test_size in test_size_list:
            for _ in range(num_of_tests):
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
                    train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
                    test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)

                    search_query = "SELECT exam_id FROM exams_info WHERE openml_id = {}".format(
                        openml_id)
                    existing_records = search(search_query)

                    existing_ids = [existing_record[0] for existing_record in existing_records]
                    exam_id = str(uuid.uuid4())
                    while exam_id in existing_ids:
                        exam_id = str(uuid.uuid4())
                    test_dir = dataset_dir + '/' + exam_id
                    os.mkdir(test_dir)
                    file_path = os.path.abspath(test_dir)
                    train.to_csv(test_dir + '/train.csv', index=False)
                    test.to_csv(test_dir + '/test.csv', index=False)

                    record = {
                        "openml_id": openml_id,
                        "exam_id": exam_id,
                        "test_size": test_size,
                        "file_path": file_path,
                        # Add more columns and corresponding values as needed
                    }
                    table_name = 'exams_info'

                    insert_record(record, table_name, database_config=config)
                except Exception:
                    print("Create training and testing data fails for OpenML dataset {}, exam ID {}!".format(openml_id,
                                                                                                             exam_id))
                    shutil.rmtree(test_dir)
                    continue


def retrieve_exam(openml_id, exam_id, database_config=config):
    search_query = "SELECT file_path FROM exams_info WHERE openml_id = {} AND exam_id = '{}'".format(openml_id, exam_id)
    records = search(search_query)
    if len(records) == 0:
        print("Record does not exist!")
        exit()
    if len(records) > 1:
        print("Primary key error!")
        exit()
    exam_dir = records[0][0]
    return exam_dir


def delete_exam(openml_id, exam_id, database_config=config, enable_cascading_delete=False):
    exam_dir = retrieve_exam(openml_id, exam_id, database_config)
    try:
        shutil.rmtree(exam_dir)
    except FileNotFoundError:
        print("File does not exist!")
    conn = mysql.connector.connect(**database_config)
    cursor = conn.cursor()
    delete_query = "DELETE FROM exams_info WHERE openml_id = {} AND exam_id = '{}'".format(openml_id, exam_id)
    cursor.execute(delete_query)
    if enable_cascading_delete:
        cascading_delete_query = "DELETE FROM evaluation_results WHERE openml_id = {} AND exam_id = '{}'"\
            .format(openml_id, exam_id)
        cursor.execute(cascading_delete_query)
    conn.commit()
    cursor.close()
    conn.close()


def search(search_query, database_config=config):
    conn = mysql.connector.connect(**database_config)
    cursor = conn.cursor()
    cursor.execute(search_query)
    records = cursor.fetchall()
    conn.close()
    cursor.close()
    return records


def insert_record(record, table_name, database_config=config):
    conn = mysql.connector.connect(**database_config)
    cursor = conn.cursor()

    record_keys = list(record.keys())
    # Construct insert query
    schema_string = ""
    for item in record_keys:
        if not record_keys.index(item) == 0:
            schema_string += ", "
        schema_string += item
    placeholder_string = ""
    for i in range(len(record_keys)):
        if not i == 0:
            placeholder_string += ", "
        placeholder_string += "%s"
    insert_query = "INSERT INTO " + table_name + " (" + schema_string + ") VALUES (" + placeholder_string + ")"

    cursor.execute(insert_query, tuple(record.values()))
    conn.commit()
    cursor.close()
    conn.close()
