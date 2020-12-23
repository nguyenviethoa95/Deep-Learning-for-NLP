"""General utility functions"""

import json
import logging 
import numpy as np


class Params():
    """
    Class that loads hyperparameter from a json file 
    """
    
    def __init__(self, json_path):
        self.update(json_path)
    
    def save(self, json_path):
        """
        Saves parameters to json file
        """
        with open(json_path,"w") as f: 
            json.dump(self.__dict__,f, indent=4)
    
    def update(self,json_path):
        with open(json_path, "w") as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instances by  'params.dict['learning_rate']'"""
        return self.dict

    def set_logger(log_path):
        """ Sets the logger to log info in terminal and file 'log_path'
        :param
            log_path: string
                where to log
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))

        # Logigng to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

def save_dict_to_json(self, d, json_path):
    """
    Saves dict of floats in json file
    :param
        d: dict
            of float-castable values (np.float, int, float, ect. )
        json_path: string
            path to json_file
    """
    with open(json_path,"w") as f:
        # We need to convert the values to float for json
        d = {k:float(v) for k,v in d.items()}
        json.dump(d, f, indent=4 )