#
# Copyright (C) 2015  Ilya Markov
#
# Full copyright notice can be found in LICENSE.
#

from __future__ import print_function

import sys

import time

from pyclick.click_models.Evaluation import LogLikelihood, Perplexity
from pyclick.click_models.UBM import UBM
from pyclick.click_models.DBN import DBN
from pyclick.click_models.SDBN import SDBN
from pyclick.click_models.DCM import DCM
from pyclick.click_models.CCM import CCM
from pyclick.click_models.CTR import DCTR, RCTR, GCTR
from pyclick.click_models.CM import CM
from pyclick.click_models.PBM import PBM
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


__author__ = 'Ilya Markov'


def get_sids(path):
    sids = set()
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            sids.add(line)
            
    return sids

if __name__ == "__main__":
    print("===============================")
    print("This is an example of using PyClick for training and testing click models.")
    print("===============================")

    if len(sys.argv) < 4:
        print("USAGE: %s <click_model> <dataset> <sessions_max>" % sys.argv[0])
        print("\tclick_model - the name of a click model to use.")
        print("\tdataset - the path to the dataset from Yandex Relevance Prediction Challenge")
        print("\tsessions_max - the maximum number of one-query search sessions to consider")
        print("")
        sys.exit(1)
        
    train_sids = get_sids("./examples/data/VK_train_sid.txt")
    test_sids = get_sids("./examples/data/VK_test_sid.txt")
    
    print(len(train_sids), len(test_sids))
    raise Exception

    click_model = globals()[sys.argv[1]]()
    search_sessions_path = sys.argv[2]
    try:
        search_sessions_num = int(sys.argv[3])
    except:
        search_sessions_num = None

    search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

    # train_test_split = int(len(search_sessions) * 0.75)
    train_sessions = list(filter(lambda session: session.task in train_sids, search_sessions))
    train_queries = Utils.get_unique_queries(train_sessions)

    test_sessions = list(filter(lambda session: session.task in test_sids, search_sessions))
    test_sessions = Utils.filter_sessions(test_sessions, train_queries)
    test_queries = Utils.get_unique_queries(test_sessions)

    print("===============================")
    print("Training on %d search sessions (%d unique queries)." % (len(train_sessions), len(train_queries)))
    print("===============================")

    start = time.time()
    click_model.train(train_sessions)
    end = time.time()
    print("\tTrained %s click model in %i secs:\n" % (click_model.__class__.__name__, end - start))

    print("-------------------------------")
    print("Testing on %d search sessions (%d unique queries)." % (len(test_sessions), len(test_queries)))
    print("-------------------------------")

    loglikelihood = LogLikelihood()
    perplexity = Perplexity()

    start = time.time()
    ll_value = loglikelihood.evaluate(click_model, test_sessions)
    end = time.time()
    print("\tlog-likelihood: %f; time: %i secs" % (ll_value, end - start))

    start = time.time()
    perp_value = perplexity.evaluate(click_model, test_sessions)[0]
    end = time.time()
    print("\tperplexity: %f; time: %i secs" % (perp_value, end - start))
