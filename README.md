# AWS environment for reinforcement learning

### Introduction
The CloudFormation script and associated code provides an environment that can be used to train and test reinforcement learning algorithms for provisioning resources based on CloudWatch metrics.

### CloudFormation script

The CloudFormation script will set up the VPC, two public subnets, an elastic load balancer (ELB) and autoscaling group (ASG) for your web server, an EC2 instance for the agent, and an EC2 instance for driving calls to the web service.

### Web Service
The web service is currently just a simple "Hello World" page, and should be replaced with something more meaningful.  The web server instances are sitting behind an ELB, and scaled using an ASG.  The max number of instances is set to 5 (modifying this is done directly in the ```awsenv/cloudopt_aws.py``` file.  Cost penalties for utilization are in ```awsenv/cloudopt.py```.

### Driver EC2 instance

The Driver instance includes very simple [code](https://gist.github.com/5f0b04f8a87eef2b2a34cacd1a07da9f.git), which will have the correct ELB DNS substituted in when starting.  This will be in the home area of the ```ec2-user``` user.  The driver is run at the command line as the ```ec2-user```:
```sh
$ ./driver/driver.py 100
```
where ```100``` is the number of iterations, and can be adjusted accordingly.  If the driver is not started before the Agent codes, then there will be no CloudWatch metrics returned and the state will return all 0's.

### Agent EC2 instance

The Agent EC2 instance automatically installs the [AWS environment](https://github.com/csgwon/AWS-RL-Env.git) to be used with this system.  The three algorithms that have been implemented are Tabular Q-learning, Deep Q-learning (DQN), and Double Dueling Deep Q Learning (D3Q).  These can be launched with the following commands at the command line:

```sh
$ cd AWS-RL-Env
$ python DQN.py
$ python D3Q.py
$ python QLearn.py
```

### Acknowledgements

The DQN and D3Q implementations were provided by [Zhiguang Wang](https://github.com/wangz813).
