import os
from concurrent import futures
import argparse
import numpy as np

# import numpy as np
from scipy.ndimage.filters import minimum_filter

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


def make_min_filter_mask(mask_path, chunks, filter_shape, out_blocks=(56, 2048, 2048), mask_key='mask', n_threads=40):
    f = z5py.File(mask_path, use_zarr_format=False)
    ds_mask = f[mask_key]
    halo = list(fshape // 2 for fshape in filter_shape)
    blocking = nifty.tools.blocking(roiBegin=[0, 0, 0],
                                    roiEnd=list(ds_mask.shape),
                                    blockShape=list(out_blocks))

    ds = f.create_dataset('min_filter_mask',
                          shape=ds_mask.shape,
                          chunks=chunks,
                          dtype=ds_mask.dtype,
                          compressor='gzip')

    def mask_block(block_id):
        print("Making min filter mask for block", block_id, '/', blocking.numberOfBlocks)
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        inner_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_roi = tuple(slice(beg, end)
                          for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        mask = ds_mask[outer_roi]
        min_filter_mask = minimum_filter(mask, size=filter_shape)
        ds[inner_roi] = min_filter_mask[local_roi]

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(mask_block, block_id) for block_id in range(blocking.numberOfBlocks)]
        [t.result() for t in tasks]


def relabel_segmentation(seg_path,
                         seg_key,
                         seg_path_out,
                         seg_key_out,
                         n_threads):
    import z5py
    ds = z5py.File(seg_path, use_zarr_format=False)[seg_key]
    shape = ds.shape

    ds_out = z5py.File(seg_path_out, use_zarr_format=False).create_dataset(seg_key_out,
                                                                           dtype='uint64',
                                                                           shape=shape,
                                                                           chunks=ds.chunks,
                                                                           compressor='gzip')

    def minmax_z(z):
        print("Finding min / max for slice", z)
        zz = slice(z, z + 1)
        seg = ds[zz]
        masked = seg[seg != 0]
        mi, ma = int(masked.min()), int(masked.max())
        return mi, ma

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(minmax_z, z) for z in range(ds.shape[0])]
        minmax = np.array([t.result() for t in tasks])

    mins = minmax[:, 0]
    maxs = minmax[:, 1]

    diffs = mins[1:] - (maxs[:-1] + 1)
    diffs = np.cumsum(diffs).astype('uint64')
    diffs = np.concatenate([np.zeros((1,), dtype='uint64'), diffs])

    def relabel_z(z):
        # print("Relabeling for slice", z)
        zz = slice(z, z + 1)
        seg = ds[zz]
        seg[seg != 0] -= diffs[z]
        ds_out[zz] = seg

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(relabel_z, z) for z in range(ds.shape[0])]
        [t.result() for t in tasks]
