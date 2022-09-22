# %%
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
# https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)  # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))

    # for logging purposes
    # print(perc)
    # print(k_cluster.cluster_centers_)

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step : int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette


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
np.random.seed(2)
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

    clt = sklearn.cluster.KMeans(n_clusters=3)
    clt.fit(slide_img_hsv.reshape(-1, 3))

    n_pixels = len(clt.labels_)
    counter = Counter(clt.labels_)  # count how many pixels per cluster
    perc = {}
    for j in range(3):
        perc[j] = counter[j] / n_pixels
    perc = dict(sorted(perc.items()))

    maxPerc = max(perc.values())
    maxColorIdx = list(perc.keys())[list(perc.values()).index(maxPerc)]
    bgColor = clt.cluster_centers_[maxColorIdx]

    thresholdRange = np.array([25, 25, 25])
    thresholdImg = cv.inRange(
        slide_img_hsv, np.abs(bgColor - thresholdRange), np.abs(bgColor + thresholdRange)
    )
    thresholdImg = cv.bitwise_not(thresholdImg)

    kernel3 = np.ones((3, 3), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)
    # thresholdImg = cv.erode(thresholdImg, kernel3, iterations=1)
    thresholdImg = cv.dilate(thresholdImg, kernel3, iterations=2)
    thresholdImg = cv.erode(thresholdImg, kernel3, iterations=2)

    opening = cv.morphologyEx(thresholdImg, cv.MORPH_OPEN, kernel3, iterations=1)
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel3, iterations=1)

    # sure background area
    sure_bg = cv.dilate(opening, kernel3, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    # grid[i].imshow(cv.cvtColor(to_draw_img, cv.COLOR_BGR2RGB))
    # grid[i].imshow(slide_img_lab[:, :, 0], cmap="gray")
    markers = cv.watershed(slide_img, markers)
    slide_img[markers == -1] = [255,0,0]
    
    grid[i].imshow(markers)
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
# %%
