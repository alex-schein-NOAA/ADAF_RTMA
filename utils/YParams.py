import argparse
from ruamel.yaml import YAML
from utils.misc_functions import to_builtin



class YParams:
    """Yaml file parser with native dot and bracket notation support"""

    def __init__(self, yaml_filename, config_name="EncDec", print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        with open(yaml_filename, "rb") as _file:
            yaml = YAML().load(_file)
            for key, val in yaml[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None
                else:
                    val = to_builtin(val) #sanitize inputs so they don't cause pickling issues with saved checkpoints

                self.params[key] = val

    def __getattr__(self, key):
        """Maps dot access to the internal dict"""
        if "params" in self.__dict__ and key in self.params:
            return self.params[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, val):
        """Maps dot assignment to the internal dict"""
        if key in ["_yaml_filename", "_config_name", "params"]:
            super().__setattr__(key, val)
        else:
            self.params[key] = val

    def __getitem__(self, key):
        """Maps bracket access"""
        return self.params[key]

    def __setitem__(self, key, val):
        """Maps bracket assignment"""
        self.params[key] = val

    def __contains__(self, key):
        return key in self.params
    
    def items(self):
        return self.params.items()

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val

    def override_from_cli(self, args):
        """
        Intercepts command-line arguments and overrides matching YAML parameters.
        """
        args_dict = vars(args) if isinstance(args, argparse.Namespace) else args
        
        for key, val in args_dict.items():
            if key in self.params and val is not None:
                self.params[key] = val

