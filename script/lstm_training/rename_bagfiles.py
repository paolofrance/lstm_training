import os
import sys
import shutil
import rospkg

# DESCRIPTION: copy and rename every '.bag' file from 'home/$USER/bag' to 'cwd/data/'
SOURCE_FOLDER = 'test/'
FOLDER = 'test/'
user = os.getenv('USER')
cwd = sys.path[0]
Source_Path = '/home/' + user + '/bag/' + SOURCE_FOLDER
Destination = rospkg.RosPack().get_path('lstm_training') + '/data/' + FOLDER


def main():

    print('destination: '+Destination)

    is_exist = os.path.exists(Destination)
    if not is_exist:
        os.mkdir(Destination)

    path_list = os.listdir(Source_Path)
    path_bag_list = [filename for filename in path_list if filename.endswith("bag")]

    for count, filename in enumerate(path_bag_list):
        dst = "trial_" + str(count+1) + ".bag"
        shutil.copyfile(os.path.join(Source_Path, filename),  os.path.join(Destination, dst))


if __name__ == '__main__':
    main()
