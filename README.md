# Out of the Box Segmentation

Single script to run 

- Inference with U-Net trained with long range affinities
- Rechunking of affinities for efficient processing
- Min filter postprocessing of mask to filter predictions with insufficient context
- Watershed on distance transform
- Multicut

resulting in a segmentation of the volume.


## Usage

Expects a single n5 file with datasets `gray` containing the 
raw data and `mask` containing a mask for parts of the data that shall be segmented.
Run complete pipelne using the `run.py` script in scripts.

```
python run.py /path/to/file.n5
```

To run only specific parts, switch off parts of the pipeline
with commandline arguments:

```
python run.py /path/to/file.n5 --inference 0 --rechunk 0 --min_filter_mask 0 --watershed 0 --multicut 1
```

This example woudl only run multicut segmentation and would only work if all previous processing steps are already done.

Argument list:

- `inference`: switch on/off affinity map inference with 1/0 (default 1)
- `rechunk`: switch on/off rechunking of affinity maps with 1/0 (default 1)
- `min_filter_mask`: Switch on/off min-filtering of the inference mask (default 1)
- `watershed`: switch on/off watershed over-segmentation with 1/0 (default 1)
- `multicut`: switch on/off multicut segmentation with 1/0 (default 1)

The script will create the following datasets in the n5 file:

- `affs_xy` / `affs_z`: The averaged in-plane affinities / between plane affinities (only nearest neighbor affinity channels).
- `affs_xy_rechunked` / `affs_z_rechunked`: The affinities, rechunked for efficient downstream processing.
- `watershed`: The watershed over-segmentation.
- `multicut`: The multicut segmentation.

## Requirements
