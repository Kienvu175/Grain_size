#%%step 1 : read image and define pixel size 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io, color, measure


img = cv2.imread('C:/Users/kienv/Downloads/1.JPG', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
pixels_to_um = 0.5 # 1 pixel = 0.5um
# plt.hist(img.flat, bins=100, range=(0,255))

#%%step 2 : denoising, if required and threshold image to separate grain from boundaries
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("Threshold image", thresh)
cv2.waitKey(0)
# %% step3 : cleanup image, and create a mask for grains
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(thresh, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)
mask = dilated ==255
# cv2.imshow("threshole img", thresh)
# cv2.imshow("Eroded Image", eroded)
# cv2.waitKey(0)

# %%step 4: label the grain
s= [[1,1,1], [1,1,1], [1,1,1]]
labeled_mask, num_labels = ndimage.label(mask, structure=s)
 
# %%
img2 = color.label2rgb(labeled_mask, bg_label =0)
cv2.imshow("color label", img2)
cv2.waitKey(0)
# %% step5 : Measuring
clusters = measure.regionprops(labeled_mask,img)

propList = [
    'Area',
    'equivalent_diameter',
    'MajorAxisLength',
    'MinorAxisLength',
    'Perimeter',
    'MinIntensity',
    'MeanIntensity',
    'MaxIntensity'
]

output_file = open('Image_measurements.csv','w')
output_file .write((","+ ','.join(propList)+'\n'))

for cluster_props in clusters:
    output_file.write(str(cluster_props['Label']))
    for i,prop in enumerate(propList):
        if(prop=='Area'):
            to_print = cluster_props[prop]*pixels_to_um**2 #convert pixel square to um
        elif(prop == 'orientation'):
            to_print = cluster_props[prop]*57.2958 #cvt to degrees from radians
        elif(prop.find('Intensity') <0):  #any props without intensity in its
            to_print = cluster_props[prop]*pixels_to_um

        else:
            to_print = cluster_props[prop]
        output_file.write(',' + str(to_print))

    output_file.write('\n')
# %%
