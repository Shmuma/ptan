import os.path
import configparser


class RunFile(configparser.ConfigParser):
    """
    Configuration for the run with way to reload options.
    """
    def __init__(self, file_name):
        super(RunFile, self).__init__()
        if not self.read(file_name):
            raise FileNotFoundError(file_name)
        self.file_name = file_name
        self.mtime = os.path.getmtime(file_name)

    def check_and_reload(self):
        mtime = os.path.getmtime(self.file_name)
        if self.mtime != mtime:
            self.clear()
            self.read(self.file_name)
