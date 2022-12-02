# pollen_id
Repository for the Pollen ID project

## Setup Instructions
### Basic Setup
1. Clone this git repo: `git clone https://github.com/beelabhmc/pollen_id`
2. Install [anaconda](https://www.anaconda.com/products/distribution)
3. Create the conda environment: `conda env create -f environment.yml` (this must be done in the `pollen_id` directory)
4. Install pytorch (see [here](https://pytorch.org/get-started/locally/)). This is not included in the `environment.yml` file because it is platform dependent.
5. Activate the conda environment: `conda activate pollen_id`

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

### Server Setup
1. Copy the model.yml file into `server/api/models` and rename it to `edge_detection_model.yml`
2. Copy your trained network `.pth` file into `server/api/models` and give it a useful name.
    - You will need to update the filename that the server reads at the bottom of [classify_pollen.py](server/api/classify_pollen.py). If you change the nextwork structure, you will also need to update the network architecture in [classify_pollen.py](server/api/classify_pollen.py) as well.
    - If you change the number of classes, you will also need to update the index to class mapping in [classify_pollen.py](server/api/utils.py)

### Running the Server
The front end of the server is hosted on github pages, and will automatically be updated on each commit
The backend ML api is contained in the `server` folder. To run the server, you will need to have the conda environment activated and then run `python server.py` from the `server` folder. The required libraries for the server are the same as in [requirements.txt](requirements.txt). PyTorch should also be installed on the server following the directions above.