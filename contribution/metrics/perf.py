import numpy as np

class Perf:

    def __init__(self, metric):
        self.metric = metric

    def score(self, contribution):
        keys = list(contribution.keys())
        max_val = abs(contribution[keys[0]][self.metric])

        for key in contribution:
            value = abs(contribution[key][self.metric])
            if value > max_val:
                max_val = value

        score = {}
        for key in contribution:
            raw_value = contribution[key][self.metric]
            score[key] = raw_value / max_val
        return score