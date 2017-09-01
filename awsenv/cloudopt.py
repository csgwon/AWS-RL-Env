import math
import numpy as np
import os
import random
import subprocess

from aws_cost_model import AWSCostModel

class AbstractEnv(object):
    def __init__(self):
        self.num_instances = 1
        self.aws_cost_model = AWSCostModel(cost_change_interval=12)
        self.counter = -1

    def set_num_instances(self, action, max):
        self.num_instances += action
        if self.num_instances < 1:
            self.num_instances = 1
        elif self.num_instances > max:
            self.num_instances = max

    def _graduated_cpu_util_cost(self, utilization):
        cost = 0.0
        if utilization > 0.9:
            cost += 10.0*self.aws_cost_model.cost_per_instance
        elif utilization > 0.8:
            cost += 5.0*self.aws_cost_model.cost_per_instance
        elif utilization > 0.7:
            cost += 3.0*self.aws_cost_model.cost_per_instance
        return cost

    def _threshold_cpu_util_cost(self, utilization, threshold = 0.8):
        cost = 0.0
        if utilization > threshold:
            cost += 3.0*self.aws_cost_model.cost_per_instance
        return cost

    def _cpu_utilization_cost(self, cpu_util):
        return self._graduated_cpu_util_cost(cpu_util)

