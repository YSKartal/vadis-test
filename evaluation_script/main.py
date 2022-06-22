import random
#from pytrec_eval import RelevanceEvaluator
import os
import json
from typing import Dict, Tuple, List
from argparse import ArgumentParser

DEFAULT_METRICS = ["map@20,50", "p@1,3,5,10,20", "r@1,3,5,10,20", "rprec"]

import re
import collections
import numpy as np

from pytrec_eval_ext import RelevanceEvaluator as _RelevanceEvaluator
from pytrec_eval_ext import supported_measures, supported_nicknames

__all__ = [
    'parse_run',
    'parse_qrel',
    'supported_measures',
    'supported_nicknames',
    'RelevanceEvaluator',
]


def parse_run(f_run):
    run = collections.defaultdict(dict)

    for line in f_run:
        query_id, _, object_id, ranking, score, _ = line.strip().split()

        assert object_id not in run[query_id]
        run[query_id][object_id] = float(score)

    return run


def parse_qrel(f_qrel):
    qrel = collections.defaultdict(dict)

    for line in f_qrel:
        query_id, _, object_id, relevance = line.strip().split()

        assert object_id not in qrel[query_id]
        qrel[query_id][object_id] = int(relevance)

    return qrel


def compute_aggregated_measure(measure, values):
    if measure.startswith('num_'):
        agg_fun = np.sum
    elif measure.startswith('gm_'):
        def agg_fun(values):
            return np.exp(np.sum(values) / len(values))
    else:
        agg_fun = np.mean

    return agg_fun(values)


class RelevanceEvaluator(_RelevanceEvaluator):
    def __init__(self, query_relevance, measures, relevance_level=1):
        measures = self._expand_nicknames(measures)
        measures = self._combine_measures(measures)
        super().__init__(query_relevance=query_relevance, measures=measures, relevance_level=relevance_level)

    def evaluate(self, scores):
        if not scores:
            return {}
        return super().evaluate(scores)

    def _expand_nicknames(self, measures):
        # Expand nicknames (e.g., official, all_trec)
        result = set()
        for measure in measures:
            if measure in supported_nicknames:
                result.update(supported_nicknames[measure])
            else:
                result.add(measure)
        return result

    def _combine_measures(self, measures):
        RE_BASE = r'{}[\._]([0-9]+(\.[0-9]+)?(,[0-9]+(\.[0-9]+)?)*)'

        # break apart measures in any of the following formats and combine
        #  1) meas          -> {meas: {}}  # either non-parameterized measure or use default params
        #  2) meas.p1       -> {meas: {p1}}
        #  3) meas_p1       -> {meas: {p1}}
        #  4) meas.p1,p2,p3 -> {meas: {p1, p2, p3}}
        #  5) meas_p1,p2,p3 -> {meas: {p1, p2, p3}}
        param_meas = collections.defaultdict(set)
        for measure in measures:
            if measure not in supported_measures and measure not in supported_nicknames:
                matches = ((m, re.match(RE_BASE.format(re.escape(m)), measure)) for m in supported_measures)
                match = next(filter(lambda x: x[1] is not None, matches), None)
                if match is None:
                    raise ValueError('unsupported measure {}'.format(measure))
                base_meas, meas_args = match[0], match[1].group(1)
                param_meas[base_meas].update(meas_args.split(','))
            elif measure not in param_meas:
                param_meas[measure] = set()

        # re-construct in meas.p1,p2,p3 format for trec_eval
        fmt_meas = set()
        for meas, meas_args in param_meas.items():
            if meas_args:
                meas = '{}.{}'.format(meas, ','.join(sorted(meas_args)))
            fmt_meas.add(meas)

        return fmt_meas

def _evaluate(
    qrels: Dict[str, Dict[str, int]], 
    results: Dict[str, Dict[str, float]], 
    measures: Dict[str, str]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        evaluator = RelevanceEvaluator(qrels, [k for _,k in measures.items()])
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
        run = json.load(f)

    with open(test_annotation_file, "r") as f:
        qrels = json.load(f)

    metrics = {}
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
    print(formatted_scores)
    
    print(metrics)
    

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "Metric1": metrics["recall@1"],
                    "Metric2": metrics["recall@3"],
                    "Metric3": metrics["recall@5"],
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
                    "Metric1": metrics["recall@1"],
                    "Metric2": metrics["recall@3"],
                    "Metric3": metrics["recall@5"],
                    "Total": random.randint(0, 99),
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
