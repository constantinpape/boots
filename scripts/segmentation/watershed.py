import sys
import luigi
import os

from mc_luigi import PipelineParameter
from mc_luigi import WsdtSegmentation


# TODO read parameter from config file
def ws_masked(path):
    ppl_params = PipelineParameter()
    ppl_params.wsdtInvert = True
    ppl_params.wsdtThreshold = .2
    ppl_params.wsdtMinSeg = 25
    ppl_params.wsdtSigSeeds = 2.6
    ppl_params.nThreads = 40

    # FIXME this is some dirty hack to trick the luigi task checker
    # to schedule our task, although the n5-path is existing
    # TODO in the long run, we need the scheduler to check for path and keys
    save_path = os.path.join(path, 'watershed')
    save_key = 'w'

    luigi.run(["--local-scheduler",
               "--pathToProbabilities", path,
               "--keyToProbabilities", "affs_xy_rechunked",
               "--pathToMask", path,
               "--keyToMask", "min_filter_mask",
               "--savePath", save_path,
               "--saveKey", save_key],
              main_task_cls=WsdtSegmentation)


if __name__ == '__main__':
    path = sys.argv[1]
    ws_masked(path)
