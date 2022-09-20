# %%
import pandas as pd
import numpy as np
import pathlib
import re
import logging
import datetime

# %%
# the local path to the folder with all the files in it
data_dir = "pollen_slides"
database_name = "database.csv"

# This will get turned into a pandas dataframe after all the files are indexed added to it
database = {
    "species": [],
    "date": [],  # the date the image was captured
    "path": [],  # the local path to the image,
    # "slide_id": [],
    "image_location": [],
    "image_depth": [],
    "image_magnification": [],
    "herbarium_specimen_id": [],
}
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
## %%

img_suffixes = [".dng", ".jpg", ".jpeg", ".png"]
match_date = re.compile(
    "\d{1,2}-\d{1,2}-\d{2}"
)  # this regex matches dates in the format "mm-dd-yy"

# Loop through all the images in the data folder
for f in pathlib.Path(data_dir).glob("**/*.*"):
    # Filter for images
    if f.suffix.lower() in img_suffixes:
        # Split the path into a list of folders
        folders = f.parent.parts

        try:
            # Ignore any images that are old (they have a yellow discoloration)
            assert folders[-1] != "Old", (
                "debug",
                f"Skipping {folders[1]} {f.name} because its in an 'Old' folder",
            )

            # Make sure path is the correct length
            assert len(folders) == 3, ("warn", f"Invalid path '{f}'")

            # Make sure the path includes exactly one date in the expected location
            date = match_date.findall(folders[2])
            assert len(date) == 1, ("warn", f"Couldn't extract date from path '{f}'")

            processed_date = pd.to_datetime(date[0])

            database["species"].append(folders[1])
            database["date"].append(processed_date)
            database["path"].append(str(f))

            # If the image is older than 2019, it has a different naming convention
            if processed_date.date() < datetime.date(2019, 1, 1):
                # Make sure the image name is in the expected format
                # A few images have a different naming convention (ex: CF071515 10X.JPG)
                # But these are also a weird color, so I think its okay to ignore them
                name_segments = f.stem.split(" ")
                assert len(name_segments) >= 2, ("warn", f"Invalid name for pre-2019 image '{f.name}'")
                try:
                    database["image_location"].append(name_segments[1][3])
                except:
                    database["image_location"].append(-1)
                try:
                    database["image_depth"].append(name_segments[1][4])
                except:
                    database["image_depth"].append('')
                # Multiply magnification by 10 to match with post-2019 images
                database["image_magnification"].append(int(name_segments[1][:2]) * 10)
                database["herbarium_specimen_id"].append("")
            else:
                name_segments = f.stem.split("_")
                assert len(name_segments) == 6, ("warn", f"Invalid name for post-2019 image '{f.name}'")
                database["image_location"].append(name_segments[4][:2])
                database["image_depth"].append(name_segments[4][2])
                database["image_magnification"].append(int(name_segments[3][3:6]))
                database["herbarium_specimen_id"].append(int(name_segments[0]))
        except AssertionError as e:
            logging_level = e.args[0][0]
            logging_message = e.args[0][1]
            if logging_level == "debug":
                logging.debug(logging_message)
            elif logging_level == "warn":
                logging.warning(logging_message)

df = pd.DataFrame(database)
# %%
df.to_csv(pathlib.Path(data_dir) / database_name,index=False)
# %%
