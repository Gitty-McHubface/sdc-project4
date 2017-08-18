import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os


def calibrate_camera(calibration_dir, file_pattern='calibration*.jpg'):
    imgpoints = []
    objpoints = []

    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for image_file in glob.glob(os.path.join(calibration_dir, file_pattern)):
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (mtx, dist)


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    mag_scaled = np.uint8(255 * magnitude / np.max(magnitude))
    
    mag_binary = np.zeros_like(mag_scaled)
    mag_binary[(mag_scaled >= mag_thresh[0]) & (mag_scaled <= mag_thresh[1])] = 1
    return mag_binary


def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    
    dir_rads = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(dir_rads)
    dir_binary[(dir_rads >= thresh[0]) & (dir_rads <= thresh[1])] = 1

    return dir_binary.astype(np.uint8)


def hls_thresh(image, hls_channel='s', thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    if hls_channel == 'h':
        channel = hls[:,:,0]
    elif hls_channel == 'l':
        channel = hls[:,:,1]
    elif hls_channel == 's':
        channel = hls[:,:,2]
    else:
        raise Exception('hls_channel must be h, l or s')

    channel_binary = np.zeros_like(channel)
    channel_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        
    return channel_binary


def rgb_thresh(image, rgb_channel='r', thresh=(0, 255)):
    if rgb_channel == 'r':
        channel = image[:,:,0]
    elif rgb_channel == 'g':
        channel = image[:,:,1]
    elif rgb_channel == 'b':
        channel = image[:,:,2]
    else:
        raise Exception('rgb_channel must be r, g or b')

    channel_binary = np.zeros_like(channel)
    channel_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1

    return channel_binary


def main():
    mtx = np.array([[  1.15158804e+03,   0.00000000e+00,   6.66167057e+02],
                    [  0.00000000e+00,   1.14506859e+03,   3.86440204e+02],
                    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    dist = np.array([[ -2.35510339e-01,  -7.90388401e-02,  -1.28492202e-03,
                       8.25970342e-05,   7.22743174e-02]])
    if not mtx.size or not dist.size:
        mtx, dist = calibrate_camera('./camera_cal')

    image = cv2.imread('./test_images/test6.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    ksize = 5
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(30, 150))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(50, 130))
    magnitude = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(30, 100))
    direction = dir_thresh(undist, sobel_kernel=ksize, thresh=(0.8, 1.2))

    h = hls_thresh(undist, 'h', thresh=(19, 30))
    s = hls_thresh(undist, 's', thresh=(160, 255))
    r = rgb_thresh(undist, 'r', (220, 255))
    c = s | r
    combined = (gradx & grady) | (magnitude & direction)

    plt.imshow(magnitude, cmap='gray')
    plt.show()

    plt.imshow(gradx & grady, cmap='gray')
    plt.show()

    plt.imshow(combined | c, cmap='gray')
    plt.show()

#    plt.imshow(gradx, cmap='gray') 
#    plt.show()
#
#    plt.imshow(grady, cmap='gray') 
#    plt.show()

#    plt.imshow(h, cmap='gray') 
#    plt.show()
#
#    plt.imshow(s, cmap='gray') 
#    plt.show()
#
#    plt.imshow(r, cmap='gray') 
#    plt.show()
#
#    plt.imshow(h | s | r, cmap='gray') 
#    plt.show()


if __name__ == '__main__':
    main()
