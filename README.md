# pollen_id
Repository for the Pollen ID project

## Setup Instructions
### Basic Setup
1. Clone this git repo: `git clone https://github.com/beelabhmc/pollen_id`
2. Install [anaconda](https://www.anaconda.com/products/distribution)
3. Create the conda environment: `conda env create -f environment.yml` (this must be done in the `pollen_id` directory)

### Data Setup & Processing
1. Download all the pollen slide images from the google drive folder (they may download as multiple .zip files, you will need to unzip and combine them into one folder manually)
2. Move those files into the `pollen_id` folder.
    - _Note:_ `pollen_id` should be the parent folder with each of the indivdual species folders directly inside of it
```
├── pollen_slides
│  ├── Acmispon glaber
│  ├── Amsinckia intermedia
│  ⋮
│  ├── Sambucus nigra
│  └── Solanum umbelliferum
```
3. Download the [`model.yml.gz`](https://github.com/opencv/opencv_extra/blob/4.x/testdata/cv/ximgproc/model.yml.gz) file (this is used for edge detection during pollen segmentation)
    - After downloading, unzip it, and make sure it is in the root `pollen_id` folder and called `model.yml`
4. Run the intake data script `python intake_data.py`
    -  This will go through all the images in the `pollen_slides` folder and create a database that categorizes them based on their folder and file name
5. Run the pollen extraction script `python extract_pollen.py`
    - This extracts the individual pollen grains from each pollen slide image and stores them in a new folder `pollen_grains`

Now you are read to run the machine learning code.