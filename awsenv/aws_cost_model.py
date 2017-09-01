import numpy as np

class AWSCostModel( object ):
    def __init__( self, cost_change_interval ):
        self.cost_change_interval = cost_change_interval
        self.max_num_instances = 10
        self.previous_num_instances = 0
        self.running_instance_matrix = np.zeros( (self.max_num_instances, cost_change_interval) )

    def set_cost( self, cost_per_instance ):
        self.cost_per_instance = float(cost_per_instance)

    def _update_matrix( self ):
        self.running_instance_matrix = np.roll(self.running_instance_matrix,1, axis=1)

    def reset( self ):
        self.running_instance_matrix = np.zeros( (self.max_num_instances, self.cost_change_interval) )

    def step( self, num_instances ):
        self._update_matrix()
        n_instances = min( self.max_num_instances, num_instances )
        if n_instances > self.previous_num_instances:
            for i in range( self.previous_num_instances, n_instances ):
                self.running_instance_matrix[i][0] = 1
            self.previous_num_instances = n_instances
        elif n_instances < self.previous_num_instances:
            self.running_instance_matrix = np.roll( self.running_instance_matrix, n_instances-self.previous_num_instances, axis=0 )
            self.running_instance_matrix[n_instances:] = 0
            self.previous_num_instances = n_instances

        return sum(self.running_instance_matrix[:,0])*self.cost_per_instance

    def step_debug( self, num_instances ):
        print(self.step( num_instances ))
        print( self.running_instance_matrix )
