# anomaly-detection

## Methods
* Autoencoder: [http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
* HLAC (Higer-order Local Auto-Correlation)
features extraction: [https://core.ac.uk/download/pdf/56664348.pdf](https://core.ac.uk/download/pdf/56664348.pdf)

## Usage
* **Data set**: 
   * Find the structure and some parameters of data set folder in `library/preprocess.py`  
   * Data type is being used here is image, so please change the function in this file if you want to train another data type.
   * Folder `hlac_test_data` just contains an image to test the HLAC features extraction step. It's not a data set for the model. 
* **Language & Library**:
   * `Python 3, tensorflow, keras, numpy, sklearn` is being used.
   * Prepare the environment as above.
   
* **Run**:
   * After preparing the environment, run any file that you want. For example: `py hlac_anomaly_detect.py`
   * Note: **JetBrains's Pycharm** maybe a friendly tool for beginner.