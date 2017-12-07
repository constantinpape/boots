import argparse
import os
from subprocess import call
from util import rechunk, str2bool


# TODO store configs for stuff in config file
# - in and output chunks, blocks
# - gpus
# - number of threads
def run(path,
        do_inference,
        do_rechunk,
        do_watershed,
        do_multicut):
    assert os.path.exists(path), "Expect N5 dataset at %s" % path

    n_threads = 40
    if do_inference:
        print("Starting inference")
        call(['python', 'inference/complete_inference.py', path])

    # TODO don't hardcode chunks
    if do_rechunk:
        out_chunks = (1, 2048, 2048)
        out_blocks = (56, 2048, 2048)

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
    parser.add_argument('--watershed', type=str2bool, default='1')
    parser.add_argument('--multicut', type=str2bool, default='1')
    args = parser.parse_args()
    return args.path, args.inference, args.rechunk, args.watershed, args.multicut


if __name__ == '__main__':
    args = parse_args()
    run(*args)
