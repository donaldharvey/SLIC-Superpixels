# coding: utf-8

import numpy as np

# See https://www.tu-chemnitz.de/etit/proaut/rsrc/neubert_protzel_superpixel.pdf for definitions
# of boundary recall and undersegmentation error.


def boundary_recall_error(test, ground_truth, distance=2):
    '''
    `test`: test superpixel boundaries: numpy matrix with 1 for boundary and 0 non_boundary.
    `ground_truth`: ground truth boundaries in same format
    `distance`: the size of the square in which to look for boundary pixels, conventionally 2
    '''
    true_pos = false_neg = 0
    for x in range(test.shape[0]):
        for y in range(test.shape[1]):
            if ground_truth[x][y]:
                # it's a boundary pixel:
                # does it exist in a square around this point in `test`?
                x_region = slice(max(0, x - distance), min(test.shape[0], x + distance + 1))
                y_region = slice(max(0, y - distance), min(test.shape[1], y + distance + 1))
                if np.count_nonzero(test[x_region, y_region]):
                    true_pos += 1
                else:
                    false_neg += 1
    return true_pos / (true_pos + false_neg)


def undersegmentation_error(test_superpixels, ground_truth_segments):
    '''
    `test_superpixels`: a 2d array with each element having a nonzero integer value
                        indicating which superpixel it belongs to.
    `ground_truth_segments`: ground truth segments in the same form.
    [n.b.: naïve and probably horribly inefficient]
    '''

    def get_superpixel(p_id):
        return np.equal(p_id, test_superpixels)

    prelim = 0

    for segment_label in np.unique(ground_truth_segments):
        # get a boolean array representing the segment
        segment = np.equal(segment_label, ground_truth_segments)

        # find the superpixels with non-empty intersections with the segment
        superpixels = np.unique(np.where(segment, test_superpixels, 0))

        for superpixel_label in superpixels:
            if superpixel_label == 0:
                # we use 0 for everything outside the segment; ignore it.
                continue
            # for each intersecting superpixel, compute p_in and p_out
            px = get_superpixel(superpixel_label)
            area_in = np.count_nonzero(np.logical_and(px, segment))
            area_out = np.count_nonzero(np.logical_and(px, np.logical_not(segment)))
            prelim += min(area_in, area_out)

    return prelim / (test_superpixels.shape[0] * test_superpixels.shape[1])


def get_bsds_data(path):
    '''Load the data from a given BSDS500 matlab file.'''
    from scipy.io import loadmat
    f = loadmat(path)
    return [{'regions': i[0][0][0], 'contours': i[0][0][1]} for i in f['groundTruth'][0]]
