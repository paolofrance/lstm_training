#!/usr/bin/env python

import numpy as np
import rospy
import rospkg


class ParamLoader:
    def __init__(self):
        self.pkgpath = rospkg.RosPack().get_path('lstm_training')
        self.data_path       = self.pkgpath + rospy.get_param('data_path')
        self.model_name_save = rospy.get_param('model_name_save')
        self.model_name_load = rospy.get_param('model_name_load')
        self.dataset_name    = rospy.get_param('dataset_name')
        self.preloaded_model      = rospy.get_param('preloaded_model')
        self.transfer_learning    = rospy.get_param('transfer_learning')
        self.use_scaler_150rec    = rospy.get_param('use_scaler_150rec')
        self.n_epochs             = rospy.get_param('n_epochs')
        self.hidden_dim           = rospy.get_param('hidden_dim')
        self.output_int_dim_param = rospy.get_param('output_int_dim_param')
        self.layer_dim            = rospy.get_param('layer_dim')
        self.batch_size = rospy.get_param('batch_size')
        self.len_seq    = rospy.get_param('input_seq_len')
        self.len_out    = rospy.get_param('output_seq_len')
        self.topics  = rospy.get_param('topics')
        self.headers = np.array( self.to_list(rospy.get_param('headers') ) )
        self.n_of_trials = rospy.get_param('n_of_trials')

        self.input_sizes = self.compute_input_size()
        self.sum_input_size = np.sum(self.input_sizes)
        self.output_dim     = self.len_out * self.input_sizes[0]
        self.output_int_dim = self.output_int_dim_param * self.input_sizes[0]

        self.lr        = rospy.get_param('lr')
        self.threshold = rospy.get_param('threshold')
        self.gamma_0   = rospy.get_param('gamma_0')
        self.gamma_fin = rospy.get_param('gamma_fin')
        self.decay     = np.geomspace(self.gamma_0 , self.gamma_fin  , self.threshold)

        rospy.loginfo("params loaded")

    def to_list(self, var):
        ret = []
        for i in var.keys():
            ret.append(var[i])
        return ret

    def compute_input_size(self):
        input_size = np.empty(len(self.headers))
        for i in np.arange(len(self.headers)):
            input_size[i] = len(self.headers[i])
        return input_size.astype(int)


if __name__ == '__main__':
    rospy.init_node('load_params', anonymous=True)
    ParamLoader()
