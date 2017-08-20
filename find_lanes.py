import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os


DEBUG = True


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
    image = cv2.GaussianBlur(image, (3, 3), 0);
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
# 1 - Simple Thresholding
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
#
# 2 - Adaptive Thresholding
#    grad_binary = cv2.adaptiveThreshold(scaled, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 4) 
#
# 3 - Otsu's Binarization
#    blur = cv2.GaussianBlur(scaled,(3,3),0)
#    ret, grad_binary = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    print('DIR RADS')
    print(dir_rads[dir_rads > np.pi / 2])
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


def warp(image):
    src = np.float32([[688, 448],
                      [1126, 720],
                      [188, 720],
                      [594, 448]])

    if DEBUG:
        plt.imshow(image, cmap='gray')
        pltsrc = src.astype(np.uint8)
        plt.plot((688, 1126), (448, 720), 'r')
        plt.plot((1126, 188), (720, 720), 'r')
        plt.plot((188, 594), (720, 448), 'r')
        plt.plot((594, 688), (448, 448), 'r')
        plt.show()

    dst = np.float32([[980, 0],
                      [980, 720],
                      [320, 720],
                      [320, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

    
def main():
    mtx = np.array([[  1.15158804e+03,   0.00000000e+00,   6.66167057e+02],
                    [  0.00000000e+00,   1.14506859e+03,   3.86440204e+02],
                    [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    dist = np.array([[ -2.35510339e-01,  -7.90388401e-02,  -1.28492202e-03,
                       8.25970342e-05,   7.22743174e-02]])
    if not mtx.size or not dist.size:
        mtx, dist = calibrate_camera('./camera_cal')

    image = cv2.imread('./test_images/test5.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    plt.imshow(undist)
    plt.show()

    ksize = 3
    gradx = abs_sobel_thresh(undist, orient='x', sobel_kernel=ksize, thresh=(30, 180))
    grady = abs_sobel_thresh(undist, orient='y', sobel_kernel=ksize, thresh=(80, 255))
    magnitude = mag_thresh(undist, sobel_kernel=ksize, mag_thresh=(60, 255))
    direction = dir_thresh(undist, sobel_kernel=ksize, thresh=(0.9, 1.3))
    h = hls_thresh(undist, 'h', thresh=(19, 30))
    s = hls_thresh(undist, 's', thresh=(160, 255))
    r = rgb_thresh(undist, 'r', (220, 255))

#    np.set_printoptions(threshold=np.nan)
#    print(gradx[((gradx > 0) & (gradx < 1)) | (gradx > 1)])
    if DEBUG:
        plt.imshow(gradx, cmap='gray')
        print('gradx')
        plt.show()

        plt.imshow(h, cmap='gray') 
        print('h')
        plt.show()

        plt.imshow(s, cmap='gray') 
        print('s')
        plt.show()

        plt.imshow(h & s, cmap='gray') 
        print('h & s')
        plt.show()

        plt.imshow(r, cmap='gray') 
        print('r')
        plt.show()

        plt.imshow(magnitude, cmap='gray') 
        print('magnitude')
        plt.show()

        plt.imshow(direction, cmap='gray') 
        print('direction')
        plt.show()

        plt.imshow(magnitude & direction, cmap='gray') 
        print('magnitude & direction')
        plt.show()

    vertices = np.array([[(650, 400),(1200, 720),(100, 720),(650, 400)]], dtype=np.int32)
    combined =  gradx | (h & s) | r | (magnitude & direction)
    
    if DEBUG:
        plt.imshow(combined, cmap='gray')
        print('combined')
        plt.show()

    masked = region_of_interest(combined, vertices)

    if DEBUG:
        print('masked')
        plt.imshow(masked, cmap='gray')
        plt.show()

    warped = warp(masked)

    print('warped')
    plt.imshow(warped, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
