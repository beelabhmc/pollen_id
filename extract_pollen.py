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
# %%
pollen_slides_dir = "pollen_slides"
pollen_slides_database_name = "database.csv"
# %%
pollen_slides_df = pd.read_csv(
    pathlib.Path(pollen_slides_dir) / pollen_slides_database_name
)
# %%
dim = 5
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
np.random.seed(92846)
chosen_idx = np.random.choice(
    pollen_slides_400x_filtered_df.shape[0], replace=False, size=dim * dim
)

img_downscale = 20

for i, (index, row) in enumerate(
    pollen_slides_400x_filtered_df.iloc[chosen_idx].iterrows()
):
    slide_img = cv.imread(row["path"])
    slide_img = cv.resize(slide_img, (int(slide_img.shape[1] / img_downscale), int(slide_img.shape[0] / img_downscale)))

    slide_img_blurred = cv.medianBlur(slide_img, 5)
    # slide_img_hsv = cv.cvtColor(slide_img, cv.COLOR_BGR2HSV)
    # slide_img_lab = cv.cvtColor(slide_img, cv.COLOR_BGR2LAB)

    image_to_detect = slide_img_blurred[:, :, 1]
    norm = np.zeros(image_to_detect.shape)
    image_to_detect = cv.normalize(image_to_detect, norm, 0, 255, cv.NORM_MINMAX)

    to_draw_img = cv.cvtColor(image_to_detect, cv.COLOR_GRAY2BGR)

    edges = skimage.feature.canny(image_to_detect, sigma=1.5)
    # result = skimage.transform.hough_ellipse(edges, threshold=4)
    # result.sort(order='accumulator')

    # result = result[-10:]

    # # Estimated parameters for the ellipse
    # for j in range(len(result)):
    #     best = list(result[j])
    #     yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    #     orientation = best[5]

    #     # Draw the ellipse on the original image
    #     # cy, cx = skimage.draw.ellipse_perimeter(yc, xc, a, b, orientation)
    #     # to_draw_img[cy, cx] = (0, 0, 255)
    #     cv.ellipse(to_draw_img, (xc, yc), (a, b), orientation, 0, 360, (0, 0, 25 5), 1)

    # Detect two radii
    hough_radii = np.arange(5, 200, 5)
    hough_res = skimage.transform.hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=40)
    for center_y, center_x, radius in zip(cy, cx, radii):
        cv.circle(to_draw_img, (center_x, center_y), radius, (0, 255, 0), 1)

    grid[i].imshow(cv.cvtColor(to_draw_img, cv.COLOR_BGR2RGB))
    # grid[i].imshow(slide_img_lab[:, :, 0], cmap="gray")
    # grid[i].imshow(edges, cmap="gray")
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
# %%
