"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID):U7016951
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01
k = 0.04
#height, width = img.shape
# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

#bw = plt.imread('0.png')
#bw = np.array(bw * 255, dtype=int)
bw = cv2.imread('Harris_3.jpg',cv2.IMREAD_GRAYSCALE)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################
def Harris_corner_response(img,k,Ix2,Ixy,Iy2):
    height, width = img.shape
    R = np.zeros((img.shape))
    for i in range(height):
        for j in range(width):
                M = np.array([[Ix2[i,j],Ixy[i,j]],[Ixy[i,j],Iy2[i,j]]])
                R[i,j] = np.linalg.det(M) - k*(np.trace(M)**2)
                
    return R

response = Harris_corner_response(bw,k,Ix2,Ixy,Iy2)
######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
def thresholding(R,threshold = 10000):
    '''
    This function take in the Harris Corner Response matrix 
    and return N corners coordinates in N*2 matrix.'''
    height,width = R.shape # take the height and width of R
    corners = [] # initial the corners data
    for i in range(height):
        for j in range(width):

            if R[i,j]>threshold:
                corners.append([i,j])

    return corners

# Display corner response


plt.subplot(1,3,3)
plt.imshow(bw,cmap = 'gray')
for coord in thresholding(response):
    #circle = matplotlib.patches.Circle(coord,10)
    plt.plot(coord[0], coord[1], color='green', marker='o')

plt.subplot(1,3,1)
plt.imshow(response,cmap = 'gray')
plt.axis('off')
plt.title('Harris Corner Response')

plt.subplot(1,3,2)
plt.imshow(bw)
plt.axis('off')
plt.title('Harris Corner Solution')

plt.show()
