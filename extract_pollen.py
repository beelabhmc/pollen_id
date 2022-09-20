# %%
import pandas as pd
import numpy as np
import pathlib
import logging
import datetime
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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

img_downscale = 6

for i, (index, row) in enumerate(
    pollen_slides_400x_filtered_df.iloc[chosen_idx].iterrows()
):
    slide_img = cv.imread(row["path"])
    slide_img = cv.resize(slide_img, (int(slide_img.shape[1] / img_downscale), int(slide_img.shape[0] / img_downscale)))

    slide_img_blurred = cv.medianBlur(slide_img, 9)
    # slide_img_hsv = cv.cvtColor(slide_img, cv.COLOR_BGR2HSV)
    # slide_img_lab = cv.cvtColor(slide_img, cv.COLOR_BGR2LAB)

    image_to_detect = slide_img_blurred[:, :, 1]
    norm = np.zeros(image_to_detect.shape)
    image_to_detect = cv.normalize(image_to_detect, norm, 0, 255, cv.NORM_MINMAX)

    to_draw_img = cv.cvtColor(image_to_detect, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(
        image_to_detect,
        cv.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=50,
        # param1=75,
        # param2=80,
        param1=10,
        param2=0.8,
        minRadius=20,
        maxRadius=0,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for j in circles[0, :]:
            # draw the outer circle
            cv.circle(to_draw_img, (j[0], j[1]), j[2], (0, 255, 0), 10)
            # draw the center of the circle
            cv.circle(to_draw_img, (j[0], j[1]), 5, (0, 0, 255), 10)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    # params = cv.SimpleBlobDetector_Params()
    
    # Set Area filtering parameters
    # params.filterByArea = True
    # params.minArea = 100
    
    # Set Circularity filtering parameters
    # params.filterByCircularity = True
    # params.minCircularity = 0.5
    
    # Set Convexity filtering parameters
    # params.filterByConvexity = True
    # params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    # detector = cv.SimpleBlobDetector_create(params)
        
    # # Detect blobs
    # keypoints = detector.detect(image_to_detect)
    # print(keypoints)
    
    # # Draw blobs on our image as red circles
    # blank = np.zeros((1, 1))
    # to_draw_img = cv.drawKeypoints(to_draw_img, keypoints, blank, (0, 0, 255),
    #                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    grid[i].imshow(cv.cvtColor(to_draw_img, cv.COLOR_BGR2RGB))
    # grid[i].imshow(slide_img_lab[:, :, 0], cmap="gray")
    # grid[i].imshow(image_to_detect, cmap="gray")
    grid[i].get_yaxis().set_ticks([])
    grid[i].get_xaxis().set_ticks([])
# %%
