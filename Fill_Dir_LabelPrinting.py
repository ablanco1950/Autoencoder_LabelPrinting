# -*- coding: utf-8 -*-
"""

 Alfonso Blanco García , November 2025
"""
import os
import re
import cv2

From_dirname = "Label Printing Defect Version 2.v25-original-images.voc\\train\\"
To_dirname = "label_data\\train\\"


ContGoodTrain=0
ContBadTrain=0

for root, dirnames, filenames in os.walk(From_dirname):
        
        
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA) # adapting any size image

                 if filename[0:3] == "BAD":
                     cv2.imwrite(To_dirname +"\\bad_label\\" + filename, image)
                     ContBadTrain=ContBadTrain+1
                 else:
                     if filename[0:3] == "GOO":
                        cv2.imwrite(To_dirname +"\\good_label\\" + filename, image)
                        ContGoodTrain=ContGoodTrain+1
# add valid registers

From_dirname = "Label Printing Defect Version 2.v25-original-images.voc\\valid\\"

for root, dirnames, filenames in os.walk(From_dirname):
        
         
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA) # adapting any size image

                 if filename[0:3] == "BAD":
                     cv2.imwrite(To_dirname +"\\bad_label\\" + filename, image)
                     ContBadTrain=ContBadTrain+1
                 else:
                     if filename[0:3] == "GOO":
                        cv2.imwrite(To_dirname +"\\good_label\\" + filename, image)
                        ContGoodTrain=ContGoodTrain+1
                        
print ("train good label " +str(ContGoodTrain))
print ("train bad label " +str(ContBadTrain))

From_dirname = "Label Printing Defect Version 2.v25-original-images.voc\\test\\"
To_dirname = "label_data\\test\\"

ContGoodTest=0
ContBadTest=0

for root, dirnames, filenames in os.walk(From_dirname):
        
         
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff|JPEG)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #image = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA) # adapting any size image

                 if filename[0:3] == "BAD":
                     cv2.imwrite(To_dirname +"\\bad_label\\" + filename, image)
                     ContBadTest=ContBadTest+1
                 else:
                     if filename[0:3] == "GOO":
                        cv2.imwrite(To_dirname +"\\good_label\\" + filename, image)
                        ContGoodTest=ContGoodTest+1

print ("test good label " +str(ContGoodTest))
print ("test bad label " +str(ContBadTest))
                        
                     
 
