import sys
import pickle
from typing import Any

import cotengra as ctg


def read_input() -> Any:
    return pickle.load(sys.stdin.buffer)


def write_output(data: Any):
    raw = pickle.dumps(data, protocol=3)
    sys.stdout.buffer.write(raw)
    sys.stdout.flush()


def filter_options(options: dict[str, Any]) -> dict[str, Any]:
    return {k:v for k, v in options.items() if v is not None}


inputs, outputs, size_dict, options = read_input()
options = filter_options(options)

opt_hq = ctg.HyperOptimizer(methods="kahypar", **options)
out = opt_hq.search(inputs, outputs, size_dict)

write_output(out.get_ssa_path())
