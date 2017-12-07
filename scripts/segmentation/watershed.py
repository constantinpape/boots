import sys
import luigi
import json

from mc_luigi import PipelineParameter
from mc_luigi import WsdtSegmentation

import numpy as np


# TODO read parameter from config file
def ws_masked(path):
    ppl_params = PipelineParameter()
    ppl_params.wsdtInvert = True
    ppl_params.wsdtThreshold = .2
    ppl_params.wsdtMinSeg = 25
    ppl_params.wsdtSigSeeds = 2.6
    ppl_params.nThreads = 40

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", path,
               "--keyToProbabilities", "affs_xy_rechunked",
               "--pathToMask", path,
               "--keyToMask", "mask"],
               main_task_cls=WsdtSegmentation)


if __name__ == '__main__':
    path = sys.argv[1]
    ws_masked(path)
