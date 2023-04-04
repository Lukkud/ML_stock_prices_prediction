import json


class Config:
    def __init__(self, file_path):
        self.config_file_path = file_path
        file = open(file_path)
        self.config_object = json.load(file)

    def parameter(self, param_name):
        try:
            param = self.config_object[param_name]
        except KeyError:
            print(f"Parameter {param_name} not defined in config.json")
            raise

        return param

    def set_param(self, param_name, param_value):
        self.config_object[param_name] = param_value
