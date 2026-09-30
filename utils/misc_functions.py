import sys
import argparse
from collections.abc import Mapping

#########################

def str2bool(value):
    """argparse type for booleans: 'False' must not become the truthy string 'False'."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("true", "t", "yes", "y", "1"):
        return True
    if lowered in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def auto_type(value):
    """argparse type for keys whose YAML value is None: None/bool/int/float, else str."""
    lowered = value.strip().lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _arg_type(example):
    if isinstance(example, bool):
        return str2bool
    if isinstance(example, int):
        return int
    if isinstance(example, float):
        return float
    if isinstance(example, str):
        return auto_type if example == "None" else str
    return auto_type


def load_config_section(config_filepath, config_name="EncDec"):
    """The named section of an ADAF YAML config as plain builtins (anchors/merge keys resolved)."""
    from ruamel.yaml import YAML

    with open(config_filepath, "rb") as f:
        return to_builtin(YAML().load(f)[config_name])


def set_user_params(parser, config_name="EncDec"):
    """Add CLI overrides for the selected YAML section and parse the arguments."""
    argv = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    known, _ = parser.parse_known_args(argv)
    config = load_config_section(known.config_filepath, config_name)

    group = parser.add_argument_group(f"config overrides (from {known.config_filepath})")
    for key, val in config.items():
        if isinstance(val, list):
            elem_type = _arg_type(val[0]) if val else auto_type
            group.add_argument(f"--{key}", type=elem_type, nargs="*", default=None)
        else:
            group.add_argument(f"--{key}", type=_arg_type(val), default=None)

    return parser.parse_args()

####

def to_builtin(value):
    """Convert ruamel container/scalar types into plain Python builtins."""
    if isinstance(value, Mapping):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_builtin(v) for v in value)

    # Preserve exact builtin scalar types
    if type(value) in (bool, int, float, str) or value is None:
        return value

    # Coerce scalar subclasses to builtins (can cause issues with saved checkpoints if not sanitized)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)

    return value