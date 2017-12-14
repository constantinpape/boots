import sys
import os
import luigi
from mc_luigi import BlockwiseMulticutSegmentation, PipelineParameter
from hashlib import md5


def multicut(path):
    ppl_params = PipelineParameter()
    # TODO change to new input file syntax
    cache_folder = os.path.join('/data/papec/cache/',
                                'cache_' + str(md5(path.encode()).hexdigest()))
    inp_file = {'data': [path, path, path],
                'cache': cache_folder,
                'seg': path,
                'keys': {'data': ['gray', 'affs_xy_rechunked', 'affs_z_rechunked'],
                         'seg': 'watershed'}}
    ppl_params.read_input_file(inp_file)

    # TODO expose some of this as parameters
    ppl_params.nThreads = 40
    ppl_params.features = ['affinitiesXY', 'affinitiesZ']
    ppl_params.zAffinityDirection = 2
    ppl_params.separateEdgeClassification = True
    ppl_params.nFeatureChunks = 120
    ppl_params.ignoreSegLabel = 0

    ppl_params.useSimpleFeatures = True

    ppl_params.multicutWeightingScheme = 'xy'
    ppl_params.multicutWeight = 15
    ppl_params.multicutBeta = 0.5

    ppl_params.subSolverType = 'kl'
    ppl_params.globalSolverType = 'kl'

    n_levels = 2

    # TODO correct path
    rf  = '/groups/saalfeld/saalfeldlab/sampleE/cremi_ABC_randomforests'

    luigi.run(['--local-scheduler',
               '--pathToSeg', path,
               '--keyToSeg', 'watershed',
               '--pathToClassifier', rf,
               '--numberOfLevels', str(n_levels),
               '--savePath', os.path.join(path, 'segmentations'),
               '--saveKey', 'multicut'],
              main_task_cls=BlockwiseMulticutSegmentation)


if __name__ == '__main__':
    path = sys.argv[1]
    multicut(path)
