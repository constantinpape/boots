import argparse
import os
from subprocess import call
from util import rechunk, str2bool, make_min_filter_mask


# TODO store configs for stuff in config file
# - in and output chunks, blocks
# - gpus
# - number of threads
def run(path,
        do_inference,
        do_rechunk,
        do_min_filter,
        do_watershed,
        do_multicut):
    assert os.path.exists(path), "Expect N5 dataset at %s" % path

    # TODO don't hardcode settings
    n_threads = 40
    out_chunks = (1, 2048, 2048)
    out_blocks = (56, 2048, 2048)
    net_in_shape = (84, 268, 268)
    net_out_shape = (56, 56, 56)

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

    if do_multicut:
        print("Starting multicut")
        call(['python', 'segmentation/multicut.py', path])


# argparser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--inference', type=str2bool, default='1')
    parser.add_argument('--rechunk', type=str2bool, default='1')
    parser.add_argument('--min_filter_mask', type=str2bool, default='1')
    parser.add_argument('--watershed', type=str2bool, default='1')
    parser.add_argument('--multicut', type=str2bool, default='1')
    args = parser.parse_args()
    return args.path, args.inference, args.rechunk, args.min_filter_mask, args.watershed, args.multicut


if __name__ == '__main__':
    args = parse_args()
    run(*args)
