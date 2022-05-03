# random-forest-for-kits2019

The original data file and feature file are too large to upload. 
The KITS19 dataset can be accessed from https://github.com/neheller/kits19.
Then they are transfered into slices through 

python conversion_data_RF.py -d ./kits19/data -o ./data

where the ./kits19/data is the folder of data downloaded from the KITS19 dataset and ./data is the folder of slices.

Then use the Feature_extraction.ipynb to extract the feature and split into training and valication set of two npy file. 

The Random_forest_v3.ipynb jupyter notebook file is the final version of training the random forest model for semantic segmantation.


![alt text](https://github.com/carlwen/random-forest-for-kits2019/blob/main/figure_1.png)
An example slice. Left: original image, middle: image after Sobel filtering, right: ground truth (green for kidney and yellow for tumor).

![alt text](https://github.com/carlwen/random-forest-for-kits2019/blob/main/figure_2.png)
An example of prediction. Left: original image of one slice after resampling, middle: ground truth (green for kidney and yellow for tumor), right: prediction).

![alt text](https://github.com/carlwen/random-forest-for-kits2019/blob/main/tabel.png)
