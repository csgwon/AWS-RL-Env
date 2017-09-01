import boto3
import datetime
import math
import numpy as np
import os
import pytz
import time
import threading

from cloudopt import AbstractEnv
from aws_cost_model import AWSCostModel

class AWSEnv(AbstractEnv):
    def __init__(self, as_group, elb, elb_url):
        super(AWSEnv, self).__init__()
        self.as_client  = boto3.client('autoscaling')
        self.elb_client = boto3.client('elb')
        self.cw_client  = boto3.client('cloudwatch')
        self.as_group   = as_group
        self.devnull    = open(os.devnull, 'w')
        self.aws_cost_model.set_cost(cost_per_instance=0.108)
        self.elb_name   = elb
        self.prev_utilization_sum = 0.0
        self.prev_network_packets_in_sum = 0
        self.prev_request_count_sum = 0
        self.prev_latency_sum = 0.0
        self.time_step = 300 # 5 min intervals on CloudWatch metrics

    def _scale_servers(self, num_instances):
        self.as_client.set_desired_capacity(AutoScalingGroupName=self.as_group, DesiredCapacity=num_instances, HonorCooldown=False)

    def _get_max_date(self, datapoints):
        dates = [x['Timestamp'] for x in datapoints]
        return max(dates)

    def _get_max_date_datapoint(self, datapoints, max_date):
        for datapoint in datapoints:
            if np.abs(datapoint['Timestamp'] - max_date) < datetime.timedelta(minutes=2):
                return datapoint
        return {'Average':0,'Sum':0}

    def _get_metrics(self):
        instance_ids = [x['InstanceId'] for x in self.elb_client.describe_instance_health(LoadBalancerName=self.elb_name)['InstanceStates']]
        end_time = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
        start_time = end_time - datetime.timedelta(minutes=10)
        utilization_sum = 0.0
        network_packets_in_sum = 0
        request_count_sum = 0
        latency_sum = 0.0
        cw_instance_count = 0 # instance count with non-null cloudwatch metrics
        max_date = None
        for id in instance_ids:
            cpu = self.cw_client.get_metric_statistics(Namespace='AWS/EC2',
                                                       MetricName='CPUUtilization',
                                                       Dimensions=[{'Name':'InstanceId', 'Value':id}],
                                                       Statistics=['Average'],
                                                       Period=300,
                                                       StartTime=start_time, EndTime=end_time)
            net_packets_in =  self.cw_client.get_metric_statistics(Namespace='AWS/EC2',
                                                                   MetricName='NetworkPacketsIn',
                                                                   Dimensions=[{'Name':'InstanceId', 'Value':id}],
                                                                   Statistics=['Average'],
                                                                   Period=300,
                                                                   StartTime=start_time, EndTime=end_time)
            if len(cpu['Datapoints']) > 0:
                max_date = self._get_max_date(cpu['Datapoints'])
                utilization_sum += self._get_max_date_datapoint(cpu['Datapoints'], max_date)['Average']
                cw_instance_count += 1
            if len(net_packets_in['Datapoints']) > 0:
                if max_date == None:
                    max_date = self._get_max_date(net_packets_in['Datapoints'])
                network_packets_in_sum += self._get_max_date_datapoint(net_packets_in['Datapoints'], max_date)['Average']

        request_count = self.cw_client.get_metric_statistics(Namespace='AWS/ELB',
                                                              MetricName='RequestCount',
                                                              Dimensions=[{'Name':'LoadBalancerName', 'Value':self.elb_name}],
                                                              Statistics=['Sum'],
                                                              Period=300,
                                                              StartTime=start_time, EndTime=end_time)
        latency =  self.cw_client.get_metric_statistics(Namespace='AWS/ELB',
                                                         MetricName='Latency',
                                                         Dimensions=[{'Name':'LoadBalancerName', 'Value':self.elb_name}],
                                                         Statistics=['Average'],
                                                         Period=300,
                                                         StartTime=start_time, EndTime=end_time)

        # use last valid values if we get nothing back
        if utilization_sum > 0:
            self.prev_utilization_sum = utilization_sum
        else:
            utilization_sum = self.prev_utilization_sum
            print('DEBUG: Using previous cpu util value')

        if network_packets_in_sum > 0:
            self.prev_network_packets_in_sum = network_packets_in_sum
        else:
            network_packets_in_sum = self.prev_network_packets_in_sum
            print('DEBUG: Using previous network packets in value')

        # get ELB metrics
        if len(request_count['Datapoints']) > 0:
            if max_date == None:
                max_date = self._get_max_date(request_count['Datapoints'])
            request_count_sum = self._get_max_date_datapoint(request_count['Datapoints'], max_date)['Sum']
            self.prev_request_count_sum = request_count_sum
        else:
            request_count_sum = self.prev_request_count_sum
            print('DEBUG: Using previous request count value')

        if len(latency['Datapoints']) > 0:
            if max_date == None:
                max_date = self._get_max_date(latency['Datapoints'])
            latency_sum = self._get_max_date_datapoint(latency['Datapoints'], max_date)['Average']
            self.prev_latency_sum = latency_sum
        else:
            latency_sum = self.prev_latency_sum
            print('DEBUG: Using previous latency value')


        if len(instance_ids) != self.num_instances:
            print('DEBUG: number of instances different from the instances in ASG: '+str(self.num_instances)+','+str(len(instance_ids)))
        return (float(utilization_sum)/cw_instance_count/100.0, float(request_count_sum)/cw_instance_count, network_packets_in_sum, latency_sum)

    def _run(self, num_instances):
        time.sleep(self.time_step)
        cost = self.aws_cost_model.step(num_instances)
        return (cost, self._get_metrics())

    def run(self, num_instances):
        self.counter += 1
        return self._run(num_instances)
    
    def step(self, action):
        super(AWSEnv, self).set_num_instances(action, max=5)
        if action != 0:
            self._scale_servers(self.num_instances)
        
        cost, observation  = self.run(self.num_instances)
        observation = (self.num_instances,)+observation
        utilization = observation[1]
        reward = -cost
        reward -= self._cpu_utilization_cost(utilization)
        done = False   # This will be continuing indefinitely, so no completion state
        info = None    
        return (observation, reward, done, info)
        
    def reset(self):
        self._scale_servers(1)
        return (self.num_instances,)+self._get_metrics()
