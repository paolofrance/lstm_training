#!/usr/bin/env python3

import numpy as np
from bagpy import bagreader
import rospkg


def load_bag_to_dataframe(path, topics):
    bag = bagreader(path)
    print(bag.topic_table)

    for t in topics:
        csv = bag.message_by_topic(t)            # return a csv


if __name__ == '__main__':

    TOPICS = ['/current_pose', '/current_velocity', '/human_wrench', '/robot_ref_pos']
    FOLDER = 'test1/'
    N_OF_TRIALS = 25

    ALL_BAGS_df = []
    for i in np.arange(1, N_OF_TRIALS + 1):
        BAG_PATH = rospkg.RosPack().get_path('lstm_training')+'/data/'+FOLDER+'trial_'+str(i)+'.bag'
        load_bag_to_dataframe(BAG_PATH, TOPICS)
