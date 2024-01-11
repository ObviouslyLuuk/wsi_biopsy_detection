import numpy as np
import cv2



"""
0/445 has a good example of a biopsy that should be split
1/445 has a nice positive control with edge artifacts, and also good biopsy examples
5/445 has good positive control example with lots of issues, 
    also artifacts that are seen as biopsies
12/445 has a lot of the same biopsies in different areas
15/445 has an air bubble artifact
16/445 has example of positive control that's small enough to be a biopsy
    also an artifact on top of biopsies
27/445 another example of artifact seen as biopsy
28/445 good example of too small correctly ignored
30/445 slide without white background, lots of double biopsies
31/445 pink artifact seen as biopsy
44/445 another example of positive control that's small enough to be a biopsy
45/445 slide that has limited spacing
46/445 star artifacts correctly ignored
48/445 one biopsy ignored because it's too small
61/445 air bubble artifact seen as biopsy
73/445 biopsy ignored because it's within another's area
"""


###############################################################################
# WholeSlideData Helpers
###############################################################################
def get_micron2pixel_fn(spacing):
    """
    Get a function that converts micrometers to pixels at a given spacing.
    Spacing is the micrometers per pixel.

    Parameters:
        spacing: float
            the micrometers per pixel

    Returns:
        function: the function that converts micrometers to pixels,
          at its given spacing, defaulting to the spacing given here.
    """
    def m2p(micrometers, spacing=spacing):
        """
        Convert micrometers to pixels, at a given spacing.
        Spacing is the micrometers per pixel.

        Parameters:
            micrometers: int / float
                the micrometers to convert
            spacing: float
                the micrometers per pixel

        Returns:
            int: the pixels at the given spacing
        """
        return int(micrometers / spacing)
    return m2p


def get_patch(wsi, x, y, w, h, spacing, cut_margin=0.02):
    """
    Get patch from wsi at coordinates x, y, w, h, which should be in terms of
    micrometers.

    Spacing is the pixels per micrometer of the patch, so if an x was found
    at a spacing of 4, then the x in terms of pixels at the wsi's spacing.
    To get the x in terms of micrometers, we need to multiply by the spacing
    before calling this function.
    WholeSlideData's get_patch function takes in x and y in terms of pixels
    at a spacing of 0.25 and w and h in terms of pixels at the given spacing.

    Parameters:
        wsi: WholeSlideImage object
            the wsi to get the patch from
        x: int / float
            left x coordinate of the patch in terms of micrometers
        y: int / float
            top y coordinate of the patch in terms of micrometers
        w: int / float
            width of the patch in terms of micrometers
        h: int / float
            height of the patch in terms of micrometers
        spacing: float
            spacing of the patch
        cut_margin: float
            percentage of the patch to cut off on each side
    """
    # Set x and y to be the center of the box
    x = x + w/2
    y = y + h/2

    # Get the new spacing
    x = x * 4
    y = y * 4
    w = (w / spacing) * (1-2*cut_margin)
    h = (h / spacing) * (1-2*cut_margin)

    return wsi.get_patch(x, y, w, h, spacing=spacing)




###############################################################################
# MASKING
###############################################################################
def get_edge_pixel_indices(img):
    '''
    Returns an array of edge pixel coordinates for a given image.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the edge pixels from.

    Returns:
        edge_pixels: np.ndarray of shape (n, 2)
    '''
    edge_pixels = np.array([np.array([i, 0]) for i in range(img.shape[0])] + \
                       [np.array([img.shape[0]-1, i]) for i in range(img.shape[1]-1, 0, -1)] + \
                       [np.array([i, img.shape[1]-1]) for i in range(img.shape[0]-1, 0, -1)] + \
                       [np.array([0, i]) for i in range(img.shape[1]-1, 0, -1)]).astype(np.uint16)
    return edge_pixels


def get_img_edges(img):
    '''
    Returns the image with only the edges.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the edges from.

    Returns:
        edges: np.ndarray of shape (n, m, 3)
    '''
    indices = get_edge_pixel_indices(img)
    return img[indices[:, 0], indices[:, 1]]


def get_most_common_color(img):
    '''
    Returns the most common color in a given image.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the most common color from.

    Returns:
        color: np.ndarray of shape (3,)
    '''
    colors, counts = np.unique(img.reshape(-1, 3), axis=0, return_counts=True)
    return colors[np.argmax(counts)]


def get_similar_color_mask(img, color, threshold=0.1):
    '''
    Returns a mask of the pixels in the image that are similar to the given color.

    Parameters:
        img: np.ndarray of shape (..., 3)
            The image to get the similar color mask from.
        color: np.ndarray of shape (3,)
            The color to compare the pixels to.
        threshold: int or float
            The threshold for the color similarity.

    Returns:
        mask: np.ndarray of shape (n, m)
            The similar color mask (True for similar, False for not similar)
    '''
    # Assert that img, color and threshold are of the correct and compatible type (int VS float)
    if type(threshold) == int:
        assert img.dtype == color.dtype == np.uint8, "Image and color must be of type np.uint8 when threshold is of type int, but are of type {} and {}.".format(img.dtype, color.dtype)
    elif type(threshold) == float:
        assert img.dtype == color.dtype == np.float32, "Image and color must be of type np.float32 when threshold is of type float, but are of type {} and {}.".format(img.dtype, color.dtype)

    return np.linalg.norm(img.astype(np.float32) - color.astype(np.float32), axis=-1) <= threshold


def get_background_mask(img, threshold=0.1, not_background_color=None, only_edges=True):
    '''
    Returns a mask of the pixels in the image that are not background.
    Whatever is the most common color in the image is considered the background color.

    Parameters:
        img: np.ndarray of shape (n, m, 3)
            The image to get the background mask from.
        threshold: int or float
            The threshold for the color similarity.
                default: 0.1 (float) or 26 (int)
        not_background_color: np.ndarray of shape (3,)
            If given, this will be excluded as a contender for the background color.
        only_edges: bool
            If True, only the edges will be considered for the most common color.
            True is also faster.
    
    Returns:
        mask: np.ndarray of shape (n, m)
            The background mask (True for foreground, False for background)
        background_color: np.ndarray of shape (3,)
            The background color.
    '''
    # Assert that img, color and threshold are of the correct and compatible type (int VS float)
    if type(threshold) == int:
        assert img.dtype == np.uint8, "Image must be of type np.uint8 when threshold is of type int."
    elif type(threshold) == float:
        assert img.dtype == np.float32, "Image must be of type np.float32 when threshold is of type float."

    # Get the most common color in the image, from the pixels we are considering
    contender_img = get_img_edges(img) if only_edges else img
    if not_background_color is not None:
        contender_img = contender_img[~get_similar_color_mask(contender_img, not_background_color, threshold)]
    background_color = get_most_common_color(contender_img)

    return ~get_similar_color_mask(img, background_color, threshold), \
        background_color


def get_contours(mask, closing_iterations=1):
    '''
    Returns the contours of a given mask.

    Parameters:
        mask: np.ndarray of shape (n, m)
            The mask to get the contours from.
        closing_iterations: int
            The number of closing iterations to do on the mask.
    
    Returns:
        contours: np.ndarray of shape (NUM_CONTOURS, NUM_POINTS, 2)
            The contours of the mask.
        hierarchy: np.ndarray of shape (NUM_CONTOURS, 4)
            The hierarchy of the contours. 
            (Next, Previous, First_Child, Parent)
        bounding_rects: list of len NUM_CONTOURS of tuples of shape (4,)
            The bounding rectangles of the contours.
            (x, y, w, h) where x and y are the top left corner coordinates
        mask_closed: np.ndarray of shape (n, m)
            The mask after closing.
    '''
    # Do closing on the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # Get the contours of the mask
    contours, hierarchy = cv2.findContours(mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c[:,0,:] for c in contours]
    hierarchy = hierarchy[0]
    return contours, hierarchy, [cv2.boundingRect(c) for c in contours], mask_closed

# MASKING FOR AREAS
####################

def get_scanned_areas_from_wsi(wsi, spacing=8, process_spacing=16,
    bg_sim_threshold=26, get_bg_by_edges=True,
    approximate_tissue_bg_color=np.array([225,225,225], dtype=np.uint8),
    mask_closing_iterations=1,
    cut_patch_margin=0.01):
    """
    Return areas from the slide that are scanned, as opposed to one solid null color.

    Parameters:
        wsi: WholeSlideImage
            the slide to get areas from
        spacing: float
            the spacing to use for the areas
        process_spacing: float
            the spacing to use for the background mask and contours
        bg_sim_threshold: int
            the color similarity threshold to use for the background mask
        get_bg_by_edges: bool
            whether to get the background by using only the edges of the slide.
            Useful when a slide has more tissue background than non-scanned slide.
        approximate_tissue_bg_color: np.array of shape (3,)
            the approximate color of the tissue background.
            Used to prevent masking out the scanned tissue background.
        mask_closing_iterations: int
            the number of iterations to use for the mask closing
        cut_patch_margin: float
            the margin to cut from the patch edges (in percent)

    Returns:
        scanned_areas: list of np.arrays
            the areas from the slide at given spacing
        img: np.array
            the slide image at process_spacing
        mask_closed: np.array
            the closed mask at process_spacing
        contours: list of np.arrays of shape (n, 1, 2)
            the contours of the tissue at process_spacing
        boxes: list of tuples
            the bounding boxes of the contours (x, y, w, h) with x, y the top left corner
            at process_spacing
    """
    # Get maximum spacing of the slide
    max_spacing = min(max(wsi.spacings), process_spacing)
    img = wsi.get_slide(max_spacing)

    mask, bg_color = get_background_mask(img, bg_sim_threshold, 
                                         approximate_tissue_bg_color, get_bg_by_edges)
    contours, hierarchy, boxes, mask_closed = get_contours(mask, mask_closing_iterations)
    scanned_areas = [get_patch(wsi,
                            *[p*max_spacing for p in [x, y, w, h]],
                            spacing, cut_patch_margin) for x, y, w, h in boxes]
    
    return scanned_areas, img, mask_closed, contours, boxes

# MASKING FOR BIOPSIES
#######################

def get_biopsy_mask(patch):
    """
    Use a blur on an average of a strict and generous mask.
    The strict mask masks out more of the background, but also more of the tissue.
    The generous mask masks out less of the background, but also less of the artifacts.

    The strict mask is dilated and the generous mask is eroded to make sure they overlap.
    Then they are averaged and blurred and clipped to get a smooth mask.
    """
    # Use a blur on an average of a strict and generous mask
    kernel = np.ones((5,5),np.uint8)
    strict_mask, bg_color = get_background_mask(patch, 10, only_edges=False)
    strict_mask = cv2.dilate(strict_mask.astype(np.uint8), kernel, iterations=1)
    generous_mask = ~get_similar_color_mask(patch, bg_color, 3)
    generous_mask = cv2.erode(generous_mask.astype(np.uint8), kernel, iterations=1)
    combo_mask = cv2.GaussianBlur((strict_mask + generous_mask).astype(np.float32),(9,9),5) / 2
    return cv2.erode(combo_mask.astype(np.float32), kernel, iterations=2) > 0.1




###############################################################################
# CONTOUR FILTERING
###############################################################################
def get_interpolations(p1, p2, weight):    
    '''
    Returns point in between two points, or an array of points in between 
    two arrays of points. If weight is 0, the first point is returned, if
    weight is 1, the second point is returned, and if weight is 0.5, the
    point halfway in between the two points is returned.

    Parameters:
        p1: np.ndarray of shape (2,) or (n, 2)
            The first point or array of points.
        p2: np.ndarray of shape (2,) or (n, 2)
            The second point or array of points.
        weight: float
            The weight of the interpolation.

    Returns:
        interpolation: np.ndarray of shape (2,) or (n, 2)
            The interpolated point or array of points.
    '''
    return p1 + (p2 - p1) * weight


# def sample_points_in_contour_area(contour, n, distance_from_edge):
#     '''
#     Returns n points sampled from the contour area, 
#     with a relative distance from the edge.

#     Parameters:
#         contour: np.ndarray of shape (n, 2)
#             The contour surrounding the area to sample points from.
#         n: int
#             The number of points to sample.
#         distance_from_edge: int or float
#             The relative distance from the edge to sample the points from.
#             0 is the edge, 1 is the center of the contour.

#     Returns:
#         points: np.ndarray of shape (n, 2)
#             The points sampled from the contour area.
#     '''
#     # Get the center of the contour
#     center = np.mean(contour, axis=0)

#     # Get random set of contour points of size n
#     n = min(n, len(contour))
#     contour = np.random.choice(contour, size=n, replace=False)

#     # Get the points in between the contour points and the center
#     return get_interpolations(contour, center, distance_from_edge).astype(np.uint32)
    

# def contour_area_is_masked(mask, contour, distance_from_edge=0.5, validate=True):
#     '''
#     Returns True if the contour area is masked, False otherwise.
#     Works by sampling points from the contour area and taking the mean
#     of the mask at those points

#     Parameters:
#         mask: np.ndarray of shape (n, m)
#             The image to get the contour area color from.
#         contour: np.ndarray of shape (n, 2)
#             The contour surrounding the area to sample points from.
#             A contour is a list of points, where each point is a list of two coordinates.
    
#     Returns:
#         masked: bool
#             True if the contour area is masked, False otherwise.
#     '''
#     # Assert that the mask is of the correct type
#     assert mask.dtype == np.uint8, "Mask must be of type np.uint8."
#     assert mask.max() == 1, "Mask must be binary."

#     # Get the points in the contour area
#     contour_area_points = sample_points_in_contour_area(contour, len(contour), distance_from_edge)

#     # Filter out points outside the contour area
#     if validate:
#         contour_area_points = contour_area_points[np.where(cv2.pointPolygonTest(contour, contour_area_points, False) >= 0)]

#     # Get the color of the contour area
#     contour_area_color = np.mean(mask[contour_area_points[:, 0], contour_area_points[:, 1]])

#     # Return True if the contour area is masked, False otherwise
#     # It is masked if the color is 0, which means it is black
#     return contour_area_color < 0.5


def get_children_per_contour(hierarchy):
    """
    Returns a dictionary mapping contour index to a list of its children.

    Parameters:
        hierarchy: list
            list of lists of 4 elements
            [next, previous, first_child, parent]

    Returns:
        children_per_contour: dictionary
            dict mapping contour index to a list of its children
    """
    children_per_contour = {}
    for i in range(len(hierarchy)):
        child = hierarchy[i][2]
        children = []
        while child != -1:
            children.append(child)
            child = hierarchy[child][0]
        children_per_contour[i] = children
    return children_per_contour


def assign_bool_to_contours(contours, children_per_contour, mask, indices, printing=False):
    """
    Assigns a boolean to each contour, indicating whether it has background or foreground pixels.
    Recursively finds most nested contours first and look at the center pixel to determine whether
    it is background or foreground, then that means the parent is the opposite.

    Parameters:
        contours: list
            List of contours
        children_per_contour: dictionary
            Dictionary mapping contour index to a list of its children
        mask: numpy array
            Mask of the background (0) and foreground (1)
        indices: list
            List of indices of contours we are interested in

    Returns:
        bg_per_contour: dictionary
            Dictionary mapping contour index to a boolean indicating whether 
            it has background or foreground pixels
    """
    bg_per_contour = {}
    def is_background(i):
        children = children_per_contour[i]
        if i in bg_per_contour:
            return bg_per_contour[i]
        elif len(children) == 0:
            # No children, so check center pixel
            center = np.mean(contours[i], axis=0).astype(np.int32)
            point = center
            # Check that center is actually inside contour
            for j in range(len(contours[i])):
                inside_contour = cv2.pointPolygonTest(contours[i][:,None,:], (int(point[0]), int(point[1])), False) > 0
                if inside_contour:
                    break
                point = get_interpolations(center, contours[i][j], np.random.rand(1).item()*.75).astype(np.int32)
            # We need to flip x and y for some reason
            bg = mask[point[1], point[0]] == 0
            if not inside_contour and printing:
                print("Center is not inside contour (contour {} of length {})".format(i, len(contours[i])))
            bg_per_contour[i] = bg
            return bg
        else:
            # Has children, so check children
            bg = not is_background(children[0])
            bg_per_contour[i] = bg
            return bg
        
    for i in indices:
        is_background(i)
    return bg_per_contour


def contour_covers_patch(contour, patch):
    """
    Returns whether a contour covers a patch.
    Takes the bounding box of the contour and checks whether it covers the patch 
    more than 90%.

    Parameters:
        contour: numpy array
            Contour
        patch: numpy array
            Patch

    Returns:
        bool
            Whether the contour covers the patch
    """
    x, y, w, h = cv2.boundingRect(contour)
    return w*h > 0.9*patch.shape[0]*patch.shape[1]


def get_contour_edge_hits(c, patch):
    """
    Returns the contour edge hit coords in a patch.

    Parameters:
        c: numpy array (n, 2)
            contour
        patch: numpy array
            Patch

    Returns:
        edge_hits: list
            List of coords of contour edge hits in a patch
    """
    return [p for p in c[:, :] if p[0] == 0 or p[0] == patch.shape[1]-1 or 
                                  p[1] == 0 or p[1] == patch.shape[0]-1]

# CONTOUR FILTER APPLICATION
#############################

def filter_indices(indices, condition):
    """
    Filters out indices of contours that do not satisfy a condition.

    Parameters:
        indices: list
            List of indices of contours we are interested in
        condition: function
            Function that takes a contour and returns a boolean

    Returns:
        indices: list
            List of indices of contours we are interested in
        filtered_indices: list
            List of indices of contours that do not satisfy the condition
    """
    filtered_indices = []
    for i in indices:
        if not condition(i):
            filtered_indices.append(i)
    indices = [i for i in indices if i not in filtered_indices]
    return indices, filtered_indices


def apply_contour_filters(contours, hierarchy, patch, mask, m2p, min_size=200):
    """
    Applies filters to contours to get the contours we are interested in.

    Parameters:
        contours: list
            List of contours
        hierarchy: list
            list of lists of 4 elements
            [next, previous, first_child, parent]
        patch: numpy array
            Patch
        mask: numpy array
            Mask of the background (0) and foreground (1)
        m2p: function
            Function that converts from micron to pixel
        min_size: int
            Minimum size of a contour expressed in micron,
            this is squared to get the minimum area in square micron

    Returns:
        indices: list
            List of indices of contours we are interested in
        filtered_indices: tuple
            Tuple of lists of indices of contours that do not satisfy the condition
            in the order: too small, background, covered, edge hit
    """
    # Filter out contours that are too small
    min_area = m2p(min_size)**2
    indices, too_small_indices = filter_indices(list(range(len(contours))), 
                lambda i: cv2.contourArea(contours[i]) >= min_area)

    # Filter out contours that are background
    children_per_contour = get_children_per_contour(hierarchy)
    bg_per_contour = assign_bool_to_contours(contours, children_per_contour, mask, indices)
    indices, bg_indices = filter_indices(indices, 
                lambda i: bg_per_contour[i] == False)

    # Filter out contours that cover the patch
    indices, covered_indices = filter_indices(indices, 
                lambda i: not contour_covers_patch(contours[i], patch))

    # Filter out contours that have more than 2 edge hits
    indices, edge_hit_indices = filter_indices(indices, 
                lambda i: len(get_contour_edge_hits(contours[i], patch)) <= 2)
    
    return indices, (too_small_indices, bg_indices, covered_indices, edge_hit_indices)




###############################################################################
# CONTOUR SPLITTING
###############################################################################
def split_contour(contour, split_idx1, split_idx2):
    '''
    Splits a contour into two closed contours at the given
    split index pair. The split index pair is a pair of indices
    that are the indices of the points in the contour that
    should be connected to form the split.

    Parameters:
        contour: np.ndarray of shape (n, 2)
            The contour to split.
        split_idx1: int
            The index of the first point in the contour to split.
        split_idx2: int
            The index of the second point in the contour to split.

    Returns:
        contour1: np.ndarray of shape (n, 2)
            The first contour.
        contour2: np.ndarray of shape (n, 2)
            The second contour.
    '''
    assert type(split_idx1) == type(split_idx2) == int, "Split index pair must be of type int."
    assert split_idx1 < split_idx2, "Split index 1 must be smaller than split index 2."

    indices_between = np.arange(split_idx1, split_idx2)
    indices_outside = np.concatenate([np.arange(0, split_idx1), np.arange(split_idx2, len(contour))])

    # Indices between are the first contour
    contour1 = contour[indices_between]

    # Indices outside are the second contour
    contour2 = contour[indices_outside]

    return contour1, contour2



# TODO: Try the idea where for each point pair you calculate euclidian distance
    # but also contour index distance, one should be low, one high
    # normalize them, and look at distribution to see if a split is likely
    # Then if it is you take the smallest euclidian distance and split there




###############################################################################
# GET POTENTIAL BIOPSIES
###############################################################################
def get_biopsies_from_scanned_area(scanned_area, m2p, min_size=300):
    """
    Returns the contours and boxes of the biopsies in a scanned area.

    Parameters:
        scanned_area: numpy array
            Scanned area
        m2p: function
            Function that converts from micron to pixel
        min_size: int
            Minimum size of a contour expressed in micron,
            this is squared to get the minimum area in square micron

    Returns:
        contours: list
            List of contours of the biopsies
        boxes: list
            List of bounding boxes of the biopsies
            (x, y, w, h) where (x, y) is the top left corner
        indices: list
            List of selected indices of contours of the biopsies
        filtered_indices: tuple
            Tuple of lists of indices of contours that do not satisfy the condition
            in the order: too small, background, covered, edge hit     
        mask: numpy array
            Processed mask of the background (0) and foreground (1)       
    """
    mask = get_biopsy_mask(scanned_area)
    contours, hierarchy, boxes, _ = get_contours(mask, 0)
    indices, filtered_indices = apply_contour_filters(
        contours, hierarchy, scanned_area, mask, m2p=m2p, min_size=min_size)
    
    return contours, boxes, indices, filtered_indices, mask
