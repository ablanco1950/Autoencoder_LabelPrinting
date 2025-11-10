# https://github.com/YoNG-Zaii/Casting-Products-Defects-Detection/blob/main/Autoencoder.ipynb
# https://www.tensorflow.org/tutorials/generative/autoencoder?hl=es-419
# Adapted and modified by Alfonso Blanco García
#
# Dataset: 
# https://universe.roboflow.com/university-science-malaysia/label-printing-defect-version-2/dataset/25
#

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model
from tensorflow.math import less

# Data Transformation

def quantify_image(img):
    # compute a greyscale histogram over an image and normalize it
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

def transform_data_from_path(imagePath):
    # convert all images in the imagePath
    # to greyscale histogram values (normalised)
    data = []
   

    # loop over the imagePath directory
    for imageName in os.listdir(imagePath):
        # load the image
        image = cv2.imread(imagePath + '\\' + imageName)
        # quantify the image and update the data list
        features = quantify_image(image)
        data.append(features)
        
        
    return np.array(data)

def transform_data_from_path_test(imagePath):
    # convert all images in the imagePath
    # to greyscale histogram values (normalised)
    data = []
    TabimageName=[]

    # loop over the imagePath directory
    for imageName in os.listdir(imagePath):
        # load the image
        image = cv2.imread(imagePath + '\\' + imageName)
        # quantify the image and update the data list
        features = quantify_image(image)
        data.append(features)
        TabimageName.append(imageName)
        
    return np.array(data), np.array(TabimageName)

# The paths to the images
defect_path = os.path.join('label_data', 'train', 'bad_label')
normal_path = os.path.join('label_data', 'train', 'good_label')

# As deep learning performs better for large datasets,
# we will use the augmented image set instead of raw image set.

defect = transform_data_from_path(defect_path)
normal = transform_data_from_path(normal_path)

print('Defect:', len(defect))
print('Normal:', len(normal))

# Defect: 178
# Normal: 458

# As we are using novelty detection method, we will train the autoencoder
# using only the normal casting. So, we will separate the normal casting from the
# defected casting.

normal_train, normal_test = train_test_split(normal, test_size=0.2, random_state=42)

defect_train, defect_test = train_test_split(defect, test_size=0.2, random_state=42)


print('Normal')
print('Train:', len(normal_train))
print('Test:', len(normal_test))

print()
print('Defective')
print('Train:', len(defect_train))
print('Test:', len(defect_test))

#Normal
#Train: 366
#Test: 92

#Defective
#Train: 142
#Test: 36



# Let's plot a normal label greyscale histogram.

plt.hist(normal_train[0])
plt.title("A Normal Label  Greyscale Histogram")
plt.show()

plt.hist(defect_train[0])
plt.title("A Defective Label Greyscale Histogram")
plt.show()


# Model Training

class NoveltyDetector(Model):
  def __init__(self):
    super(NoveltyDetector, self).__init__()
    self.encoder = Sequential([
      layers.Dense(128, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
    ])

    self.decoder = Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(256, activation="sigmoid")
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = NoveltyDetector()
autoencoder.compile(optimizer='adam', loss='mae')

# The autoencoder is trained using only the normal label ,
# but is evaluated using the full test set.

test = np.concatenate((normal_test, defect_test), axis=0)

# True means normal label while False means defective label
test_labels = np.full(len(test), True, dtype=bool)
test_labels[-len(defect_test):] = False

print("")
print(len(test))

# 128



history = autoencoder.fit(normal_train, normal_train, 
          epochs=30, 
          batch_size=32,
          validation_data=(test, test),
          shuffle=False,
          verbose=0)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

encoded_data = autoencoder.encoder(normal_test).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

# Plot and compare the original and reconstructed first image for normal label
plt.plot(normal_test[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(256), decoded_data[0], normal_test[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("Reconstruction Effort for Normal Label")
plt.show()

# It seems like the reconstruction error for one normal casting image is low.
# Let's do the same for defective label image.

encoded_data = autoencoder.encoder(defect_test).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

# Plot and compare the original and reconstructed first image for defective casting
plt.plot(defect_test[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(256), decoded_data[0], defect_test[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.title("Reconstruction Effort for Defect Label")
plt.show()

#In contrast, the reconstruction error for one defective casting image is higher.
# We can use this observation to detect defective label.

# Detect Anomalies/Novelty
# We will detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold.
# For this, we will calculate the mean average error for normal samples from the training set,
# then classify future examples as anomalous (defective) if the reconstruction error is higher than
# one standard deviation from the training set.

reconstructions = autoencoder.predict(normal_train)
train_loss = losses.mae(reconstructions, normal_train)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()


threshold = np.mean(train_loss) + np.std(train_loss)

print("")
print("Threshold: ", threshold)
#Threshold:  0.002980382196929098
print("")

reconstructions = autoencoder.predict(defect_test)
test_loss = losses.mae(reconstructions, defect_test)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()



# The reconstruction errors for the defective test samples are concentrated between 0.012 and 0.019.

reconstructions = autoencoder.predict(test)
loss = losses.mae(reconstructions, test)
preds = less(loss, threshold)

print(classification_report(test_labels, preds, target_names=['Defective', 'Normal']))

# Compared to all other models, we can see that autoencoder model achieves a high accuracy of 0.91.
# The recalls for both defective and normal casting are also high, at 0.99 and 0.82 respectively.

# Now, we will try the threshold values in the range of [0.01, 0.013).

#optimal_th = 0 # MOD
optimal_th = threshold # MOD

max_accuracy = -1
#for threshold in np.arange(0.01, 0.013, 0.0001): # MOD
Cont=0
for threshold in np.arange(optimal_th, optimal_th + 0.013, 0.0001):    # MOD
    #print(threshold)
    Cont=Cont+1
    if Cont > 100:break
    #print(threshold)
    preds = less(loss, threshold)
    accuracy = accuracy_score(test_labels, preds)
    if(accuracy > max_accuracy):
        max_accuracy = accuracy
        optimal_th = threshold
    print(f'threshold: {threshold: .4f}, accuracy: {accuracy}')

print("")    
print(f'Best threshold: {optimal_th: .4f}, accuracy: {max_accuracy}')
print("")

# Best threshold:  0.0047, accuracy: 1.0.



preds = less(loss, optimal_th)
print(classification_report(test_labels, preds, target_names=['Defective', 'Normal']))


#print(test_labels)
#print("======================")
#print(preds)
#print("======================")


# TRUE TEST with data that has not been in the process trained

# The paths to the images
defect_path_true_test = os.path.join('label_data', 'test', 'bad_label')
normal_path_true_test = os.path.join('label_data', 'test', 'good_label')

# As deep learning performs better for large datasets,
# we will use the augmented image set instead of raw image set.

defect_true_test, nameImage_defect_true_test = transform_data_from_path_test(defect_path_true_test)
normal_true_test, nameImage_normal_true_test = transform_data_from_path_test(normal_path_true_test)

print('Defect:', len(defect_true_test))
print('Normal:', len(normal_true_test))

reconstructions = autoencoder.predict(defect_true_test)
test_loss = losses.mae(reconstructions, defect_true_test)
preds = less(test_loss, threshold)

#print("************************************")
#print(preds)

print("")
print (" LIST OF LABEL PRINTING ")
print("")

for i in range(len(nameImage_defect_true_test)):
    
    if str(preds[i]) == "tf.Tensor(True, shape=(), dtype=bool)":
    
       print( nameImage_defect_true_test[i] + " NORMAL" )
    else:
        print( nameImage_defect_true_test[i] + " DEFECT" )

reconstructions = autoencoder.predict(normal_true_test)
test_loss = losses.mae(reconstructions, normal_true_test)
preds = less(test_loss, threshold)

#print("+++++++++++++++++++++++++++++")
#print(preds)

for i in range(len(nameImage_normal_true_test)):
    
    if str(preds[i]) == "tf.Tensor(True, shape=(), dtype=bool)":
    
       print( nameImage_normal_true_test[i] + " NORMAL" )
    else:
        print( nameImage_normal_true_test[i] + " DEFECT" )







                










