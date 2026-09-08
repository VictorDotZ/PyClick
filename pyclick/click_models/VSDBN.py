from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List

import math

from pyclick.click_models.ClickModel import ClickModel
from pyclick.click_models.Param import ParamMLE
from pyclick.click_models.ParamContainer import QueryDocumentParamContainer


__author__ = "Victor Zenin"


@dataclass
class Ratio:
    numerator: float = 1
    denominator: float = 2


@dataclass
class ViewInfo:
    stop_view_time: List[float] = field(default_factory=list)
    view_time: List[float] = field(default_factory=list)


class VSDBN(ClickModel):
    def __init__(self):
        self._attr = defaultdict(lambda: defaultdict(Ratio))
        self._lambda = defaultdict(lambda: defaultdict(Ratio))
        self._time = defaultdict(lambda: defaultdict(ViewInfo))

        self._beta = defaultdict(lambda: defaultdict(float))

        self.max_steps = 10

        self._eps = 10**-6

    def f(self, query, document, beta_u):
        total_time = sum(self._time[query][document].view_time)
        tmp = self._eps

        for T_i in self._time[query][document].stop_view_time:
            tmp += (T_i * math.pow(math.e, -beta_u * T_i)) / (
                1.0 - math.pow(math.e, -beta_u * T_i) + self._eps
            )

        return tmp - total_time

    def H(self, query, document, beta_u):
        tmp = self._eps
        for T_i in self._time[query][document].stop_view_time:
            tmp -= (T_i * T_i * math.pow(math.e, -beta_u * T_i)) / (
                math.pow(1.0 - math.pow(math.e, -beta_u * T_i), 2.0) + self._eps
            )

        return tmp

    def optimize(self, query, document):
        beta_n = 1.0
        step = 0

        while True:
            beta_n_plus_one = beta_n - self.f(query, document, beta_n) / self.H(
                query, document, beta_n
            )
            step += 1

            if math.fabs(beta_n_plus_one - beta_n) < self._eps or step > self.max_steps:
                return beta_n_plus_one

            beta_n = beta_n_plus_one

        return beta_n

    def train(self, search_sessions):
        for session in search_sessions:
            query = session.query
            results = session.web_results
            last_click_rank = session.get_last_click_rank()

            for rank in range(last_click_rank + 1):
                document = results[rank]

                # Вычисляем привлекательность a_u
                self._attr[query][document].denominator += 1

                if document.click:
                    self._attr[query][document].numerator += 1

                # вычисляем lambda_u
                self._lambda[query][document].numerator += 1

                if document.view_time > 0.0:
                    self._lambda[query][document].denominator += document.view_time

                # собираем статистику для beta_u

                if rank != last_click_rank and document.view_time > 0.0:
                    self._time[query][document].view_time.append(document.view_time)

                if rank == last_click_rank and document.view_time > 0.0:
                    self._time[query][document].stop_view_time.append(
                        document.view_time
                    )

        for query in self._time:
            for document in self._time[query]:
                self._beta[query][document] = self.optimize(query, document)

    def get_conditional_click_probs(self, search_session):
        return
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, result in enumerate(search_session.web_results):
            attr = session_params[rank][self.param_names.attr].value()
            sat = session_params[rank][self.param_names.sat].value()

            if result.click:
                click_prob = attr * exam
                exam = 1 - sat
            else:
                click_prob = 1 - attr * exam
                exam *= (1 - attr) / click_prob

            click_probs.append(click_prob)

        return click_probs

    def get_full_click_probs(self, search_session):
        return
        session_params = self.get_session_params(search_session)
        exam = 1
        click_probs = []

        for rank, session_param in enumerate(session_params):
            attr = session_param[self.param_names.attr].value()
            sat = session_param[self.param_names.sat].value()

            click_probs.append(attr * exam)
            exam *= (1 - sat) * attr + (1 - attr)

        return click_probs

    def predict_relevance(self, query, search_result):
        return
        attr = self.params[self.param_names.attr].get(query, search_result).value()
        sat = self.params[self.param_names.sat].get(query, search_result).value()
        return attr * sat
