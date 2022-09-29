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
import logging
from tqdm.autonotebook import tqdm

# %%
pollen_slides_dir = "pollen_slides"
pollen_slides_database_name = "database.csv"
pollen_grains_dir = "pollen_grains"
run_on_full_dataset = True
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
# Filter out any 100x images
pollen_slides_400x_filtered_df = pollen_slides_df[
    pollen_slides_df["image_magnification"] == 400
]

# If we're not running on all the images, setup graphing code
if not run_on_full_dataset:
    dim = 5
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(dim, dim),  # creates 2x2 grid of axes
    )

    np.random.seed(3)
    chosen_idx = np.random.choice(
        pollen_slides_400x_filtered_df.shape[0], replace=False, size=dim * dim
    )

# how much to scale images down by for detection (if this number is changed the algorithm will need to be re-tuned)
img_downscale = 5

edge_detector = cv.ximgproc.createStructuredEdgeDetection("model.yml")

images_to_run_on = (
    pollen_slides_400x_filtered_df  # normally loop through all images
    if run_on_full_dataset  # unless the user said to run on a subset (for testing)
    else pollen_slides_400x_filtered_df.iloc[chosen_idx]
).groupby(
    ["species", "image_location", "image_depth"]
)  # TODO: Add slide ID to this groupby

# %%
# This loop could definitely be parallelized for a large speed increase
# But it takes < 10 mins to run on ~750 images on my laptop so I don't think its worth it
for group in tqdm(images_to_run_on, total=len(images_to_run_on)):
    all_contours = []
    for i, (index, row) in enumerate(group[1].iterrows()):
        # Read and resize the image
        slide_img_full_res = cv.imread(row["path"])
        slide_img = cv.resize(
            slide_img_full_res,
            (
                int(slide_img_full_res.shape[1] / img_downscale),
                int(slide_img_full_res.shape[0] / img_downscale),
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
            ((edges**2) * 255).astype(np.uint8),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )
        # draw the contours on a copy of the original image
        image_with_contours = image_to_detect.copy()
        cv.drawContours(image_with_contours, contours, -1, (255, 0, 0), 2)

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
        try:
            cv.grabCut(
                image_to_detect,
                trimap,
                rect,
                bgdModel,
                fgdModel,
                5,
                cv.GC_INIT_WITH_MASK,
            )
        except:
            logging.warning(f"Skipping... grabCut failed on image: {row['path']}")
            continue

        # create mask again
        mask2 = np.where(
            (trimap == cv.GC_FGD) | (trimap == cv.GC_PR_FGD), 255, 0
        ).astype("uint8")
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

        to_draw_img = slide_img.copy()
        # Draw all contours on the image (these are drawn in blue, the ones we use will be drawn in green later)
        for c in contours:
            cv.drawContours(to_draw_img, c, -1, (255, 0, 0), 5)

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

        if len(contours_filtered) == 0:
            logging.warning(f"Skipping... no contours found on image: {row['path']}")
            continue

        max_contour_area = max([cv.contourArea(c) for c in contours_filtered])
        contours_final = []
        for c in contours_filtered:
            if not (cv.contourArea(c) > max_contour_area * 0.1):
                continue
            contours_final.append(c)
            cv.drawContours(to_draw_img, c, -1, (0, 0, 0), 5)

        all_contours.extend(contours_final)

        if not run_on_full_dataset:
            grid[i].imshow(cv.cvtColor(to_draw_img, cv.COLOR_BGR2RGB))
            # grid[i].imshow(thresholdImg)
            grid[i].get_yaxis().set_ticks([])
            grid[i].get_xaxis().set_ticks([])

    all_contours_combined = []
    images_in_group = [(p, cv.imread(p)) for p in group[1]["path"].values]

    if len(group[1]) > 1:
        for c in all_contours:
            not_overlapping = True
            for c_included in all_contours_combined:
                c_mask = np.zeros(images_in_group[0][1].shape[:2])
                c_included_mask = c_mask.copy()

                cv.fillPoly(c_mask, [c], 1)
                cv.fillPoly(c_included_mask, [c_included], 1)

                c_mask_sum = c_mask.sum()
                c_included_mask_sum = c_included_mask.sum()
                overlapp = (c_mask * c_included_mask).sum()

                if overlapp / c_mask_sum > 0.5:
                    not_overlapping = False
                    break

            if not_overlapping:
                all_contours_combined.append(c)
    else:
        all_contours_combined = all_contours

    for i, c in enumerate(all_contours_combined):
        prefix = "train" if i >= np.ceil(len(group[1]) * 0.8) else "test"
        # TODO: Add slide id
        basePath = (
            pathlib.Path(pollen_grains_dir)
            / prefix
            / row["species"]
            / row["image_location"]
            / str(i)
        )
        basePath.mkdir(parents=True, exist_ok=True)

        for j, (p, img) in enumerate(images_in_group):
            padding = 10
            x, y, w, h = cv.boundingRect(c)

            # Pick the largest axis to make them squares
            if w > h:
                y -= (w - h) // 2
                h = w
            else:
                x -= (h - w) // 2
                w = h

            # Add padding and scale up to full resolution images
            y1 = (y - padding) * img_downscale
            y2 = (y + h + padding) * img_downscale
            x1 = (x - padding) * img_downscale
            x2 = (x + w + padding) * img_downscale

            # Move the crop back into the image if it went out of bounds
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if y2 > img.shape[0]:
                y1 -= y2 - img.shape[0]
                y2 = img.shape[0]
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if x2 > img.shape[1]:
                x1 -= x2 - img.shape[1]
                x2 = img.shape[1]

            pollen_grain = img[y1:y2, x1:x2]

            cv.imwrite(
                str((basePath / f"{row['image_depth']}.png")), 
                pollen_grain,
            )

# %%
