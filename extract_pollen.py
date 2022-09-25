# %%
import pandas as pd
import numpy as np
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage
import skimage.feature
import skimage.transform
import skimage.draw
import skimage.segmentation
import scipy
# %%
pollen_slides_dir = "pollen_slides"
pollen_slides_database_name = "database.csv"
# %%
pollen_slides_df = pd.read_csv(
    pathlib.Path(pollen_slides_dir) / pollen_slides_database_name
)
# %%
# From: https://www.codepasta.com/computer-vision/2019/04/26/background-segmentation-removal-with-opencv-take-2.html
def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 50:
            break
        lastMedian = median
        median = cv.medianBlur(edgeImg, 3)


# %%
dim = 5
fig = plt.figure(figsize=(10.0, 10.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(dim, dim),  # creates 2x2 grid of axes
    # axes_pad=0.1,  # pad between axes in inch.
)

# Filter out any 100x images
pollen_slides_400x_filtered_df = pollen_slides_df[
    pollen_slides_df["image_magnification"] == 400
]
np.random.seed(3)
chosen_idx = np.random.choice(
    pollen_slides_400x_filtered_df.shape[0], replace=False, size=dim * dim
)

img_downscale = 5

edge_detector = cv.ximgproc.createStructuredEdgeDetection("model.yml")

for i, (index, row) in enumerate(
    pollen_slides_400x_filtered_df.iloc[chosen_idx].iterrows()
):
    slide_img = cv.imread(row["path"])
    slide_img = cv.resize(
        slide_img,
        (
            int(slide_img.shape[1] / img_downscale),
            int(slide_img.shape[0] / img_downscale),
        ),
    )

    # Normalizes the image (https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd)
    # I don't fully know how this works, but it effectively increases local contrast
    norm = np.zeros(slide_img.shape)
    slide_img_normalized = cv.normalize(slide_img, norm, 0, 255, cv.NORM_MINMAX)

    # This remaps the range of values in an image to fill the full 0-255 range
    slide_img_normalized = (
        slide_img_normalized - np.min(slide_img_normalized, axis=(0, 1))
    ) / (
        np.max(slide_img_normalized, axis=(0, 1))
        - np.min(slide_img_normalized, axis=(0, 1))
    )
    slide_img_normalized = (slide_img_normalized * 255).astype(np.uint8)
    image_to_detect = slide_img_normalized

    # Run an random forest edge detector
    edges = edge_detector.detectEdges(image_to_detect.astype(np.float32) / 255.0)
    filterOutSaltPepperNoise(edges)

    # Find the contours on the edges
    contours, hierarchy = cv.findContours(
        ((edges**2) * 255).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    # draw the contours on a copy of the original image
    cv.drawContours(image_to_detect, contours, -1, (255, 0, 0), 2)

    # Turn the contours into a black and white mask
    mask = np.zeros_like(edges)
    cv.fillPoly(mask, contours, 255)
    mapFg = cv.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    trimap = np.copy(mask).astype(np.uint8)
    trimap[mask == 0] = cv.GC_BGD
    trimap[mask == 255] = cv.GC_PR_BGD
    trimap[mapFg == 255] = cv.GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, slide_img.shape[1], slide_img.shape[0])
    cv.grabCut(
        image_to_detect, trimap, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK
    )

    # create mask again
    mask2 = np.where((trimap == cv.GC_FGD) | (trimap == cv.GC_PR_FGD), 255, 0).astype(
        "uint8"
    )
    slide_img_masked = slide_img * mask2[:, :, np.newaxis]

    kernel3 = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)

    # # General erosion and dilation to remove noise
    thresholdImg = mask2
    thresholdImg = cv.erode(thresholdImg, kernel3, iterations=1)
    thresholdImg = cv.dilate(thresholdImg, kernel3, iterations=2)
    thresholdImg = cv.erode(thresholdImg, kernel3, iterations=2)

    # Fill any small holes
    opening = cv.morphologyEx(thresholdImg, cv.MORPH_OPEN, kernel3, iterations=1)
    # Remove any small specs
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel3, iterations=2)

    # Watershed Code
    dist = scipy.ndimage.distance_transform_edt(opening)
    peak_idx = skimage.feature.peak_local_max(
        dist, min_distance=10, threshold_rel=0.5, labels=opening
    )

    local_max = np.zeros_like(dist, dtype=bool)
    local_max[tuple(peak_idx.T)] = True

    labels = scipy.ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    markers = skimage.segmentation.watershed(-dist, labels, mask=opening)

    # Convert the marker data into a format findContours accepts
    markers_rounded = markers.astype(np.uint8)
    contours = []
    for j in range(1, markers_rounded.max() + 1):
        c, _ = cv.findContours(
            np.array(markers_rounded == j).astype(np.uint8),
            cv.RETR_LIST,
            cv.CHAIN_APPROX_NONE,
        )
        contours += c

    # Draw all contours on the image (these are drawn in blue, the ones we use will be drawn in green later)
    for c in contours:
        cv.drawContours(slide_img, c, -1, (255, 0, 0), 5)

    contours_filtered = []
    for c in contours:
        # Fit a circle to the contour. If the contour doesn't fill 40% of the circle, skip it
        _, r = cv.minEnclosingCircle(c)
        if cv.contourArea(c) / (np.pi * r**2) < 0.4:
            continue

        # If the of the contour is less than 100 pixels, skip it
        if cv.contourArea(c) < 100:
            continue

        contours_filtered.append(c)

    max_contour_area = max([cv.contourArea(c) for c in contours_filtered])
    for c in contours_filtered:
        if not (cv.contourArea(c) > max_contour_area * 0.1):
            continue
        cv.drawContours(slide_img, c, -1, (0, 0, 0), 5)

    grid[i].imshow(cv.cvtColor(slide_img, cv.COLOR_BGR2RGB))
    # grid[i].imshow(thresholdImg)
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
# %%
