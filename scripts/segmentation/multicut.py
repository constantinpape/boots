import sys
import os
import luigi
from mc_luigi import BlockwiseMulticutSegmentation, PipelineParameter
from hashlib import md5


def multicut(path, use_simple_feats=True):
    ppl_params = PipelineParameter()
    # TODO change to new input file syntax
    cache_folder = os.path.join('/data/papec/cache/',
                                'cache_' + str(md5(path.encode()).hexdigest()))
    inp_file = {'data': [path, path, path],
                'cache': cache_folder,
                'seg': path,
                'keys': {'data': ['gray', 'affs_xy_rechunked', 'affs_z_rechunked'],
                         'seg': 'segmentation/watershed'}}
    ppl_params.read_input_file(inp_file)

    # TODO expose some of this as parameters
    ppl_params.nThreads = 40
    ppl_params.features = ['affinitiesXY', 'affinitiesZ']
    ppl_params.zAffinityDirection = 2
    ppl_params.separateEdgeClassification = True
    ppl_params.nFeatureChunks = 120
    ppl_params.ignoreSegLabel = 0

    ppl_params.useSimpleFeatures = use_simple_feats

    ppl_params.multicutWeightingScheme = 'xy'
    ppl_params.multicutWeight = 15
    ppl_params.multicutBeta = 0.5

    ppl_params.subSolverType = 'kl'
    ppl_params.globalSolverType = 'kl'

    n_levels = 2

    rf  = '/groups/saalfeld/saalfeldlab/sampleE/cremi_ABC_randomforests' if use_simple_feats \
        else '/groups/saalfeld/saalfeldlab/sampleE/cremi_ABC_randomforests_more_features'

    # dirty hack because gpu 2 does not find /groups/saalfeld/saalfeldlabe
    if not os.path.exists(rf):
        rf = '/groups/saalfeld/home/papec/cremi_ABC_randomforests_more_features'

    save_path = os.path.join(path,
                             'segmentations' if use_simple_feats else 'segs')
    print("saving multicut segmentation to", save_path)
    save_key = 'multicut' if use_simple_feats else 'multicut_more_features'

    luigi.run(['--local-scheduler',
               '--pathToSeg', path,
               '--keyToSeg', 'watershed',
               '--pathToClassifier', rf,
               '--numberOfLevels', str(n_levels)],
              main_task_cls=BlockwiseMulticutSegmentation)


if __name__ == '__main__':
    path = sys.argv[1]
    use_simple_feats = bool(int(sys.argv[2]))
    multicut(path, use_simple_feats)
