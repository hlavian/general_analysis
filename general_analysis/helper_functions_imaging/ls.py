import numpy as np
from split_dataset import SplitDataset, EmptySplitDataset, save_to_split_dataset
from general_analysis.helper_functions_imaging.general_imaging import normalize_traces
from lotr.data_preprocessing.traces import preprocess_traces


def filter_and_detrend(dataset, output_dir, fs, block_size=120, filter=True, zscore=True, win=5, verbose=True):
    # prepare the destination
    '''
    new_dataset = EmptySplitDataset(
        root=output_dir,
        name="aligned",
        shape_full=dataset.shape,
        shape_block=(block_size,) + dataset.shape_block[1:],
    )
    '''
    n_t, n_planes, x_pix, y_pix = dataset.shape

    if verbose:
        print("Normalizing...")

    norm_stack = np.zeros((n_t, n_planes, x_pix*y_pix))
    for z in range(n_planes):
        print(z)
        img = dataset[:, z, :, :]
        traces = img.reshape(img.shape[0], -1)

        # Normalize
        if zscore:
            traces = normalize_traces(traces)

        # Detrend and filter
        if filter:
            norm_stack[:, z, :] = preprocess_traces(traces.T, fn=fs).T

    if verbose:
        print("Saving...")
    img_processed = norm_stack.reshape(dataset.shape)
    save_to_split_dataset(img_processed, output_dir, block_size=block_size)

    return z#new_dataset.finalize()


