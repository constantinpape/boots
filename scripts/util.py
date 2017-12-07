import os
from concurrent import futures

import z5py
# we need nifty for the blocking
import nifty


def rechunk(in_path,
            out_path,
            in_key,
            out_key,
            out_chunks,
            out_blocks,
            n_threads):
    assert os.path.exists(in_path)
    f_in = z5py.File(in_path, use_zarr_format=False)
    ds_in = f_in[in_key]
    shape = ds_in.shape

    f_out = z5py.File(out_path, use_zarr_format=False)
    # TODO enable blosc compression in n5
    compressor = 'gzip'
    ds_out = f_out.create_dataset(out_key,
                                  shape=shape,
                                  dtype=ds_in.dtype,
                                  chunks=out_chunks,
                                  compressor=compressor)
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(shape),
                                    blockShape=list(out_blocks))

    def write_block(block_id):
        print("Writing block", block_id, "/", blocking.numberOfBlocks)
        block = blocking.getBlock(block_id)
        start, stop = block.begin, block.end
        roi = tuple(slice(sta, sto) for sta, sto in zip(start, stop))
        ds_out[roi] = ds_in[roi]

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(write_block, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result() for t in tasks]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
