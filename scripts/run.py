import argparse
import os
from subprocess import call
from hashlib import md5
from shutil import rmtree

from util import rechunk, str2bool, make_min_filter_mask, relabel_segmentation


# TODO store configs for stuff in config file
# - in and output chunks, blocks
# - gpus
# - number of threads
def run(path,
        do_inference,
        do_rechunk,
        do_min_filter,
        do_watershed,
        do_relabel,
        do_multicut):
    assert os.path.exists(path), "Expect N5 dataset at %s" % path

    # TODO don't hardcode settings
    n_threads = 40
    out_chunks = (1, 2048, 2048)
    out_blocks = (56, 2048, 2048)
    net_in_shape = (84, 268, 268)
    net_out_shape = (56, 56, 56)
    # simple features
    use_simple_feats = False

    if do_inference:
        print("Starting inference")
        call(['python', 'inference/complete_inference.py', path])

    if do_rechunk:
        # rechunk xy-affinities
        print("Starting rechunk affinities xy")
        rechunk(path, path,
                'affs_xy', 'affs_xy_rechunked',
                out_chunks, out_blocks,
                n_threads)

        # rechunk z-affinities
        print("Starting rechunk affinities z")
        rechunk(path, path,
                'affs_z', 'affs_z_rechunked',
                out_chunks, out_blocks,
                n_threads)

    if do_min_filter:
        print("Making min filter mask")
        # our shape for the min filter is the contect of the neural network
        # which is (in_shape - out_shape) // 2
        filter_shape = tuple((ins - outs) // 2 for ins, outs in zip(net_in_shape, net_out_shape))
        # increase filter to 2 * context_size + 1
        # (necessary for all predictions to be valid)
        filter_shape = tuple(2 * fshape + 1 for fshape in filter_shape)
        print("With filter shape", filter_shape)
        make_min_filter_mask(path,
                             filter_shape=filter_shape,
                             out_blocks=out_blocks,
                             chunks=out_chunks,
                             n_threads=n_threads,
                             mask_key='mask')

    if do_watershed:
        print("Starting watershed")
        call(['python', 'segmentation/watershed.py', path])

    if do_relabel:
        print("Starting relabeling")
        relabel_segmentation(path, 'watershed/w', path, 'watershed_rel', n_threads)
        rmtree(os.path.join(path, 'watershed'))
        os.rename(os.path.join(path, 'watershed_rel'),
                  os.path.join(path, 'watershed'))

    if do_multicut:
        print("Starting multicut")
        call(['python', 'segmentation/multicut.py', path,
              "1" if use_simple_feats else "0"])
        cache_folder = os.path.join('/data/papec/cache/',
                                    'cache_' + str(md5(path.encode()).hexdigest()))
        # copy segmentation to target folder and rechunk
        seg_path = None
        for f in os.listdir(cache_folder):
            if f.startswith("BlockwiseMulticutSegmentation"):
                seg_path = os.path.join(cache_folder, f)
                break
        assert seg_path is not None
        rechunk(seg_path,
                os.path.join(path, 'segmentation'),
                'multicut' if use_simple_feats else 'multicut_more_features',
                (26, 256, 256),
                (26, 1024, 1024),
                n_threads)
        # delete cache
        rmtree(cache_folder)


# argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--inference', type=str2bool, default='1')
    parser.add_argument('--rechunk', type=str2bool, default='1')
    parser.add_argument('--min_filter_mask', type=str2bool, default='1')
    parser.add_argument('--watershed', type=str2bool, default='1')
    parser.add_argument('--relabel', type=str2bool, default='0')
    parser.add_argument('--multicut', type=str2bool, default='1')
    args = parser.parse_args()
    return args.path, args.inference, args.rechunk, args.min_filter_mask, args.watershed, args.relabel, args.multicut


if __name__ == '__main__':
    args = parse_args()
    run(*args)
