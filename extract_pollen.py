# %%
from ntpath import join
import pandas as pd
import numpy as np
import pathlib
import logging
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import skimage
import skimage.feature
import skimage.transform
import skimage.draw
import sklearn.cluster
import skimage.segmentation
import scipy
from collections import Counter

# %%
pollen_slides_dir = "pollen_slides"
pollen_slides_database_name = "database.csv"
# %%
pollen_slides_df = pd.read_csv(
    pathlib.Path(pollen_slides_dir) / pollen_slides_database_name
)
# %%
dim = 2
fig = plt.figure(figsize=(10.0, 10.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(dim, dim),  # creates 2x2 grid of axes
    # axes_pad=0.1,  # pad between axes in inch.
)

pollen_slides_400x_filtered_df = pollen_slides_df[
    pollen_slides_df["image_magnification"] == 400
]
np.random.seed(3)
chosen_idx = np.random.choice(
    pollen_slides_400x_filtered_df.shape[0], replace=False, size=dim * dim
)

img_downscale = 10

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

    norm = np.zeros(slide_img.shape)
    slide_img_normalized = cv.normalize(slide_img, norm, 0, 255, cv.NORM_MINMAX)
    slide_img_blurred = cv.medianBlur(slide_img_normalized, 1)
    slide_img_blurred = cv.pyrMeanShiftFiltering(slide_img_blurred, 11, 25)
    # image_to_detect = slide_img_blurred[:, :, 1]
    image_to_detect = slide_img_blurred

    slide_img_hsv = cv.cvtColor(image_to_detect, cv.COLOR_BGR2HSV)
    # slide_img_lab = cv.cvtColor(image_to_detect, cv.COLOR_BGR2LAB)
    # to_draw_img = cv.cvtColor(image_to_detect, cv.COLOR_GRAY2BGR)

    # Find the 3 most common colors in the image
    clt = sklearn.cluster.KMeans(n_clusters=3)
    clt.fit(slide_img_hsv.reshape(-1, 3))

    n_pixels = len(clt.labels_)
    counter = Counter(clt.labels_)  # count how many pixels per cluster
    perc = {}
    for j in range(3):
        perc[j] = counter[j] / n_pixels
    perc = dict(sorted(perc.items()))

    # Figure out which of those three is most common
    maxPerc = max(perc.values())
    maxColorIdx = list(perc.keys())[list(perc.values()).index(maxPerc)]
    # Assume that that color is the background color
    bgColor = clt.cluster_centers_[maxColorIdx]

    # Threshold the image to isolate pollen from the background color
    # Note: Because we're inverting the binary image, adjustting thresholdRange has the opposite effect you would expect
    thresholdRange = np.array([25, 25, 25])
    thresholdImg = cv.inRange(
        slide_img_hsv, np.abs(bgColor - thresholdRange), np.abs(bgColor + thresholdRange)
    )
    thresholdImg = cv.bitwise_not(thresholdImg)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)

    # General erosion and dilation to remove noise
    # thresholdImg = cv.erode(thresholdImg, kernel3, iterations=1)
    thresholdImg = cv.dilate(thresholdImg, kernel3, iterations=2)
    thresholdImg = cv.erode(thresholdImg, kernel3, iterations=2)

    # Fill any small holes
    opening = cv.morphologyEx(thresholdImg, cv.MORPH_OPEN, kernel3, iterations=1)
    # Remove any small specs
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel3, iterations=2)

    # Watershed Code
    dist = scipy.ndimage.distance_transform_edt(opening)
    peak_idx = skimage.feature.peak_local_max(dist, min_distance=10, threshold_rel=0.5, labels=opening)

    local_max = np.zeros_like(dist, dtype=bool)
    local_max[tuple(peak_idx.T)] = True

    labels = scipy.ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    markers = skimage.segmentation.watershed(-dist, labels, mask=opening)

    # # Convert the marker data into a format findContours accepts
    markers_rounded = markers.astype(np.uint8)
    contours = []
    for j in range(1, markers_rounded.max() + 1):
        c, _ = cv.findContours(np.array(markers_rounded == j).astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        contours += c

    # Draw all contours on the image (these are drawn in blue, the ones we use will be drawn in green later)
    for c in contours:
        cv.drawContours(slide_img, c, -1, (255, 0, 0), 2)

    contours_filtered = []
    for c in contours:
        # Fit a circle to the contour. If the contour doesn't fill 40% of the circle, skip it
        _, r = cv.minEnclosingCircle(c)
        if cv.contourArea(c) / (np.pi*r**2) < 0.4:
            continue
        
        # If the of the contour is less than 100 pixels, skip it
        if cv.contourArea(c) < 100:
            continue

        contours_filtered.append(c)


    max_contour_area = max([cv.contourArea(c) for c in contours_filtered])
    for c in contours_filtered:
        if not (cv.contourArea(c) > max_contour_area * 0.1):
            continue
        cv.drawContours(slide_img, c, -1, (0, 255, 0), 2)
    
    grid[i].imshow(cv.cvtColor(slide_img, cv.COLOR_BGR2RGB))
    # grid[i].imshow(markers)
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
# %%
