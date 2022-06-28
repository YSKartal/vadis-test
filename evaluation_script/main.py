import random
import subprocess
import sys
#import ranxy
#from .pytrec_eval import RelevanceEvaluator
#from .ranxy import Qrels, Run, evaluate
import os
os.system('pip install ranx')

import ranx
    
#import pytrec_eval
import os
import json
from typing import Dict, Tuple, List
from argparse import ArgumentParser

from sklearn.metrics import precision_score, recall_score, f1_score
from argparse import ArgumentParser
import os
import json
from typing import List

DEFAULT_METRICS = ["map@20,50", "p@1,3,5,10,20", "r@1,3,5,10,20", "rprec"]
METRICS = {"f1": f1_score, "p": precision_score, "r": recall_score}
DEFAULT_METRICS = ["f1@macro", "p@macro", "r@macro"]

def _evaluate(
    qrels: Dict[str, Dict[str, int]], 
    results: Dict[str, Dict[str, float]], 
    measures: Dict[str, str]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, [k for _,k in measures.items()])
        scores = evaluator.evaluate(results)

        metrics = {_k:0.0 for _k,__ in scores[list(scores.keys())[0]].items()}

        for _,query_scores in scores.items():
            for score_name,score in query_scores.items():
                metrics[score_name] += score

        for k,v in metrics.items():
            metrics[k] = round(v/len(scores),5)

        return metrics

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    
    with open(user_submission_file, "r") as f:
        pred_json = json.load(f)

    with open(test_annotation_file, "r") as f:
        true_json = json.load(f)

    pred_keys = sorted(list(pred_json.keys()))
    true_keys = sorted(list(true_json.keys()))

    assert pred_keys == true_keys

    pred = [pred_json[k] for k in pred_keys]
    true = [true_json[k] for k in true_keys]

    metrics = {}
    for m in DEFAULT_METRICS:
        parts = m.split("@")
        if len(parts) == 2:
            name = ""
            avg = None

            try:
                name, avg = parts[0].lower(), parts[1]
            except:
                print(f"Invalid format for metric: {m}")

            if name in METRICS:
                score = METRICS[name](y_pred=pred, y_true=true, average=avg, zero_division=False)
                metrics[name] = score

        elif len(parts) == 1:
            name = m.lower()

            if name in METRICS:
                score = METRICS[name](y_pred=pred, y_true=true, average=avg, zero_division=False)
                metrics[name] = score
    
    print(metrics)

    
    '''metrics = {}
    for m in DEFAULT_METRICS:
        parts = m.split("@")
        if len(parts) == 2:
            name = ""
            ks = None

            try:
                name, ks = parts[0].lower(), parts[1]
            except:
                print(f"Invalid format for metric: {m}")

            if name in PYTREC_METRIC_MAPPING:
                _metric = PYTREC_METRIC_MAPPING[name]+"."+ks
                metrics[name] = _metric

        elif len(parts) == 1:
            name = m.lower()

            if name in PYTREC_METRIC_MAPPING:
                _metric = PYTREC_METRIC_MAPPING[name]
                metrics[name] = _metric

    scores = _evaluate(qrels, run, metrics)
    formatted_scores = {k.replace("_","@"): v for k,v in scores.items()}
    print(formatted_scores)'''
    
    print(metrics)
    

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "Metric1": metrics["f1"],
                    "Metric2": metrics["p"],
                    "Metric3": metrics["r"],
                    "Total": random.randint(0, 99),
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": {
                    "Metric1": random.randint(0, 99),
                    "Metric2": random.randint(0, 99),
                    "Metric3": random.randint(0, 99),
                    "Total": random.randint(0, 99),
                }
            },
            {
                "test_split": {
                    "Metric1": metrics["f1"],
                    "Metric2": metrics["p"],
                    "Metric3": metrics["r"],
                    "Total": random.randint(0, 99),
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
