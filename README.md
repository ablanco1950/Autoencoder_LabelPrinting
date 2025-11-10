# Autoencoder_LabelPrinting
Project to detect errors in label printing using an autoencoder.

Use the Roboflow file as both the training and test files:

https://universe.roboflow.com/university-science-malaysia/label-printing-defect-version-2/dataset/25

Installation:

The usual modules for a development environment are assumed to be installed: numpy, matplotlib, scikit-learn and tensorflow. Otherwise, they can be installed with a simple pip command.

Download the project to disk and download the Roboflow dataset to the same directory.

A folder named Label Printing Defect Version 2.v25-original-images.voc should have been created with the subfolders train, valid, and test.

Create the directory of folders required by the autoencoder:

python Create_Dir_LabelPrinting.py

Fill the folders in the directory with the data from the Roboflow file:

python Fill_Dir_LabelPrinting.py

Train and test the project.

Python Autoencoder_LabelPrinting.py

The console displays the results of all training processes, a training summary, and finally, a list of image names indicating whether they were detected as defective or normal, highlighting 5 label images:

BAD_ROI_160_jpg.rf.07d2fe6f9dfca8db9e09f81d52650886.jpg NORMAL
BAD_ROI_177_jpg.rf.a864443760bdd8ca91582788a5dfd81f.jpg NORMAL
BAD_ROI_182_jpg.rf.9c00f50315205fc3a08b083482cad72a.jpg NORMAL
BAD_ROI_193_jpg.rf.704ba2523011c856f589886f62d87263.jpg NORMAL
BAD_ROI_199_jpg.rf.eb16200394f11d7f75d4e8d4014d67c7.jpg NORMAL
BAD_ROI_91_jpg.rf.2fbffb57fe38ba1736d418cfec39a694.jpg NORMAL

These are assumed to be erroneous, as their names should have identified them as defective. The test is performed using a test file that was not part of the training and consists of 112 images, resulting in a 95.5% accuracy rate, consistent with the predictions obtained during training.

Notes:

The autoencoder project distorts the input images and then measures the effort required to reconstruct them, which is greater for defective images.

It is necessary to download the Roboflow label printing file as indicated, although the download page states that the images have not been preprocessed or enlarged (perhaps for this reason). Roboflow maintains many label printing files that do not yield good results when processed.

Acknowledgments and citations:

The training program is an adaptation of the one found at: https://github.com/YoNG-Zaii/Casting-Products-Defects-Detection/blob/main/Autoencoder.ipynb

It has been adapted for label defect detection, and the optimal threshold detection loop has been modified. A list of images has been added at the end, indicating whether they were detected as normal or defective, allowing for targeted correction.

https://www.tensorflow.org/tutorials/generative/autoencoder?hl=es-419

https://universe.roboflow.com/university-science-malaysia/label-printing-defect-version-2/dataset/25
