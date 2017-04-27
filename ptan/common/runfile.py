import configparser
import collections


class RunFile(configparser.ConfigParser):
    """
    Configuration for the run
    """
    def __init__(self, file_name):
        super(RunFile, self).__init__()
        if not self.read(file_name):
            raise FileNotFoundError(file_name)
