import os
import json
import random
import argparse
import numpy as np
from mturk_cores import MTurkManager, print_log
from datetime import date
from datetime import datetime


def main(args):

    # ===============================
    # step1: Parsing Configurations
    # ===============================
    human_eval_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    html_file_folder = os.path.join(human_eval_root_dir, "user_interface", "human_study")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    """
    To test HIT in sandbox, set args.test_hit = True (default);
    If post HIT for production, please set args.test_hit = False.
    """
    config_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "config_sandbox.json") if args.test_hit \
                else os.path.join(human_eval_root_dir, "mturk_experiment", "config.json")

    with open(config_dir, "r") as read_file:
        config = json.load(read_file)
    print_log("INFO", f" ====== Display Your Configuration ====== \n{config}")



    # ===============================
    # step2: MTurkManager Client Setup
    # ===============================
    """
    client functions to view:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html#MTurk.Client.create_worker_block
    """
    mturk_manager = MTurkManager(config)
    mturk_manager.set_environment()
    mturk_manager.setup_client()
    client = mturk_manager.client
    print(client.get_account_balance()['AvailableBalance'])



    # ===============================
    # step3: MTurk Tasks  --- Create HIT
    # ===============================
    """
    Step1: Retrieve Recruited Workers
    """
    recruiting_worker_file = "MTurk_results_10_11_2021_10_30.json"  ###  This recruiting_worker_file should be identical to the result_json_file in recruit_participants folder (stage1).
    recruiting_worker_file_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "recruit_participants")
    with open(os.path.join(recruiting_worker_file_dir, recruiting_worker_file), "r") as read_file:
        recruited_worker_json_file = json.load(read_file)
    

    worker_group_dict = {}
    batch_number, group_number = 20, 10
    recruited_workers = list(recruited_worker_json_file.values())[0]["Correct_Workers"]

    random.shuffle(recruited_workers)
    worker_groups = [recruited_workers[i::group_number] for i in range(group_number)]
    assert len(worker_groups) == group_number

    for g in range(group_number):
        worker_group_dict[g] = {}
        worker_group_dict[g]["WorkerIDs"] = recruited_workers[g::group_number]


    """
    Step2: Create 10 Qualification Types - For 10 Worker Groups
    """
    for g in range(group_number):
        try:   ### In case the qualification is created ###
            response = client.create_qualification_type(
                Name = f'WorkerGroup{g}',
                Description = f'The workers are in Group{g}',
                QualificationTypeStatus='Active'
            )
        except:
            pass

    QualificationTypeIds = {}
    response = client.list_qualification_types(MustBeRequestable=True, MustBeOwnedByCaller=True)['QualificationTypes']
    for r in response:
        QualificationTypeIds[r['Name']] = r['QualificationTypeId']


    """
    Step3: Associate Workers with Qualification Types
    """
    for g in range(group_number):
        qualificationyypeID = QualificationTypeIds[f'WorkerGroup{g}']
        worker_group_dict[g]['qualificationyypeID'] = qualificationyypeID
        worker_group = worker_groups[g]
        for worker_id in worker_group:
            response = client.associate_qualification_with_worker(
                QualificationTypeId= qualificationyypeID,
                WorkerId=worker_id,
                IntegerValue=1234,
                SendNotification=False
            )

    save_results_dir = os.path.join(current_dir, f"MTurk_WorkerGroup_Track.json")
    with open(save_results_dir, 'w') as json_file:
        json.dump(worker_group_dict, json_file, indent = 2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_hit', action='store_true', dest='test_hit', default=True)
    args = parser.parse_args()
    main(args)





