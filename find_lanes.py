import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import glob
import os


DEBUG_THRESHOLDING = False
DEBUG_FIT_LINES = False
DEBUG_PROCESS_IMAGE = False

# Run the pipeline on a test image.
TEST_IMAGE = None#'./test_images/straight_lines1.jpg'

# Run the pipeline on a video if TEST_IMAGE is None
IN_VIDEO = './project_video.mp4'
OUT_VIDEO = './processed_video.mp4'
CALIBRATION_DIR = './camera_cal'

MASK_ROI = np.array([[(650, 400), (1200, 720), (100, 720), (650, 400)]], dtype=np.int32)

# Saved camera distortion correction values. 
USE_SAVED_CORRECTION = True
SAVED_MTX = np.array([[ 1.15158804e+03, 0.00000000e+00, 6.66167057e+02],
                      [ 0.00000000e+00, 1.14506859e+03, 3.86440204e+02],
                      [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
SAVED_DIST = np.array([[ -2.35510339e-01, -7.90388401e-02, -1.28492202e-03,
                         8.25970342e-05, 7.22743174e-02]])

WARP_ORIG_SRC = np.float32([[684, 448],
                            [1126, 720],
                            [188, 720],
                            [595, 448]])
WARP_OVERHEAD_DST = np.float32([[960, 0],
                                [960, 720],
                                [320, 720],
                                [320, 0]])

# Scale (m / pixel) for determining lane curvature and offset
X_SCALE = 3.7 / 700
Y_SCALE = 32 / 720

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

def calibrate_camera(calibration_dir, file_pattern='calibration*.jpg',
                     row_corners=6, col_corners=9):
    imgpoints = []
    objpoints = []

    objp = np.zeros((row_corners * col_corners, 3), np.float32)
    objp[:,:2] = np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2)

    for image_file in glob.glob(os.path.join(calibration_dir, file_pattern)):
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (col_corners, row_corners), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints,
                                             gray.shape[::-1], None, None)
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

#   1 - Simple Thresholding
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
#
#   2 - Adaptive Thresholding
#    grad_binary = cv2.adaptiveThreshold(scaled, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY_INV, 5, 4) 
#
#   3 - Otsu's Binarization
#    blur = cv2.GaussianBlur(scaled,(3,3),0)
#    ret, grad_binary = cv2.threshold(blur, 0, 1,
#                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


def warp(image, src, dst):
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]),
                                 flags=cv2.INTER_LINEAR)

    if DEBUG_THRESHOLDING:
        plt.imshow(image, cmap='gray')
        plt.plot((src[0][0], src[1][0]), (src[0][1], src[1][1]), 'r')
        plt.plot((src[1][0], src[2][0]), (src[1][1], src[2][1]), 'r')
        plt.plot((src[2][0], src[3][0]), (src[2][1], src[3][1]), 'r')
        plt.plot((src[3][0], src[0][0]), (src[3][1], src[0][1]), 'r')
        plt.show()

        plt.imshow(warped, cmap='gray')
        plt.plot((dst[0][0], dst[1][0]), (dst[0][1], dst[1][1]), 'r')
        plt.plot((dst[1][0], dst[2][0]), (dst[1][1], dst[2][1]), 'r')
        plt.plot((dst[2][0], dst[3][0]), (dst[2][1], dst[3][1]), 'r')
        plt.plot((dst[3][0], dst[0][0]), (dst[3][1], dst[0][1]), 'r')
        plt.show()

    return warped


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # Defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # Defining a 3 channel or 1 channel color to fill the mask with depending on
    # the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def image_threshold(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 180))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(80, 255))
    magnitude = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(60, 255))
    direction = dir_thresh(image, sobel_kernel=ksize, thresh=(0.9, 1.3))
    h = hls_thresh(image, 'h', thresh=(16, 30))
    s = hls_thresh(image, 's', thresh=(90, 255))
    r = rgb_thresh(image, 'r', (220, 255))
    g = rgb_thresh(image, 'g', (200, 255))

    if DEBUG_THRESHOLDING:
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

        plt.imshow(g, cmap='gray') 
        print('g')
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

    combined =  gradx | (h & s) | r | g | (magnitude & direction)
    
    return combined


from collections import deque

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.reset()

    def reset(self, num_keep_recent=5):
        self.detected = False
        # Number of consecutive frames the line was not detected
        self.frames_not_detected = 0
        # X values of recent fits of the line
        self.recent_x_fitted = deque(maxlen=num_keep_recent)
        # Average x values of the fitted line over recent iterations
        self.best_x = np.array([])
        # Polynomial coefficients averaged over recent iterations
        self.best_fit = np.array([])
        # Polynomial coefficients for the most recent fit
        self.current_fit = np.array([])
        # Radius of curvature of the line in some units
        self.radius_of_curvature = None
        # Distance in meters of vehicle center from the line
        self.line_base_pos = None
        # X values for detected line pixels
        self.all_x = None
        # Y values for detected line pixels
        self.all_y = None


def sliding_window_search(overhead_image, nwindows=15, margin=80, minpix=200, left_base_range=(220, 420), right_base_range=(860, 1060)):
    # Create an output image to draw on and  visualize the result
    if DEBUG_FIT_LINES:
        out_img = np.dstack((overhead_image, overhead_image, overhead_image)) * 255

    histogram = np.sum(overhead_image[overhead_image.shape[0] // 2:,:], axis=0)

    # These will be the starting point for the left and right lines
    left_x_base = np.argmax(histogram[left_base_range[0]:left_base_range[1]]) + left_base_range[0]
    right_x_base = np.argmax(histogram[right_base_range[0]:right_base_range[1]]) + right_base_range[0]
    
    # Set height of windows
    window_height = np.int(overhead_image.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = overhead_image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated for each window
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = overhead_image.shape[0] - (window + 1) * window_height
        win_y_high = overhead_image.shape[0] - window * window_height
        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin
        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Draw the windows on the visualization image
        if DEBUG_FIT_LINES:
            cv2.rectangle(out_img, (win_x_left_low, win_y_low),
                          (win_x_left_high, win_y_high), GREEN, 2)
            cv2.rectangle(out_img, (win_x_right_low, win_y_low),
                          (win_x_right_high, win_y_high), GREEN, 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_x_left_low) &
                          (nonzero_x < win_x_left_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_x_right_low) &
                           (nonzero_x < win_x_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            left_x_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:        
            right_x_current = np.int(np.mean(nonzero_x[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds] 
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds] 

    # Left lane pixels red, right lane pixels blue
    if DEBUG_FIT_LINES:
        out_img[left_y, left_x] = list(RED)
        out_img[right_y, right_x] = list(BLUE)
        plt.imshow(out_img)

    return (left_x, left_y, right_x, right_y)


def margin_search(overhead_image, left_fit, right_fit, margin=60):
    nonzero = overhead_image.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_lane_inds = ((nonzero_x > (left_fit[0] * (nonzero_y ** 2) +
                                    left_fit[1] * nonzero_y + left_fit[2] - margin)) &
                      (nonzero_x < (left_fit[0] * (nonzero_y ** 2) +
                                    left_fit[1] * nonzero_y + left_fit[2] + margin))) 
    right_lane_inds = ((nonzero_x > (right_fit[0] * (nonzero_y ** 2) +
                                     right_fit[1] * nonzero_y + right_fit[2] - margin)) &
                       (nonzero_x < (right_fit[0] * (nonzero_y ** 2) +
                                     right_fit[1] * nonzero_y + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    left_x = nonzero_x[left_lane_inds]
    left_y = nonzero_y[left_lane_inds] 
    right_x = nonzero_x[right_lane_inds]
    right_y = nonzero_y[right_lane_inds]

    if DEBUG_FIT_LINES:
        plot_y = np.linspace(0, overhead_image.shape[0] - 1, overhead_image.shape[0])
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]

        # Color in left and right line pixels
        out_img = np.dstack((overhead_image, overhead_image, overhead_image)) * 255
        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = list(RED)
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = list(BLUE)

        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([left_fit_x + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(
            np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([right_fit_x + margin, plot_y])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), GREEN)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), GREEN)
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(result)

    return (left_x, left_y, right_x, right_y)


def get_line_curvature(line_x, line_y, y_pos, x_scale=1, y_scale=1):
    # Fit polynomial to scaled x, y
    line_fit = np.polyfit(line_y * y_scale, line_x * x_scale, 2)

    # Calculate the radii of curvature at the bottom of the image
    line_curverad = ((1 + (2 * line_fit[0] * y_pos * y_scale +
                      line_fit[1]) ** 2) ** 1.5) / np.absolute(2 * line_fit[0])

    return line_curverad


def find_lane_lines(overhead_image, left_line, right_line, max_frames_not_found=5):
    if ( left_line.frames_not_detected > max_frames_not_found or
         right_line.frames_not_detected > max_frames_not_found ):
        left_line.reset()
        right_line.reset() 

    if not (left_line.best_fit.size and right_line.best_fit.size):
        left_x, left_y, right_x, right_y = sliding_window_search(overhead_image)
    else:
        left_x, left_y, right_x, right_y = margin_search(overhead_image, left_line.best_fit,
                                                         right_line.best_fit)

    if not (left_x.size and right_x.size):
        left_line.detected = False
        right_line.detected = False
        left_line.frames_not_detected += 1
        right_line.frames_not_detected += 1
        return

    left_line.all_x = left_x
    left_line.all_y = left_y
    right_line.all_x = right_x
    right_line.all_y = right_y

    # Fit a second order polynomial to each line
    left_line.current_fit = np.polyfit(left_y, left_x, 2)
    right_line.current_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values
    y_size = overhead_image.shape[0]
    x_size = overhead_image.shape[1]
    plot_y = np.linspace(0, y_size - 1, y_size)

    left_x_fitted = ( left_line.current_fit[0] * plot_y ** 2 +
                      left_line.current_fit[1] * plot_y +
                      left_line.current_fit[2] )
    right_x_fitted = ( right_line.current_fit[0] * plot_y ** 2 +
                       right_line.current_fit[1] * plot_y +
                       right_line.current_fit[2] )

    if DEBUG_FIT_LINES:
        plt.plot(left_x_fitted, plot_y, color='yellow')
        plt.plot(right_x_fitted, plot_y, color='yellow')
        plt.xlim(0, x_size)
        plt.ylim(y_size, 0)
        plt.show()

    dist_top = right_x_fitted[0] - left_x_fitted[0]
    dist_bot = right_x_fitted[y_size - 1] - left_x_fitted[y_size - 1]
    if left_x_fitted[0] > right_x_fitted[0] or abs(dist_top - dist_bot) > 200:
        left_line.detected = False
        right_line.detected = False
        left_line.frames_not_detected += 1
        right_line.frames_not_detected += 1
        return

    # X values of each line
    left_line.recent_x_fitted.append(left_x_fitted)
    right_line.recent_x_fitted.append(right_x_fitted)

    left_line.best_x = np.average(left_line.recent_x_fitted, axis=0)
    right_line.best_x = np.average(right_line.recent_x_fitted, axis=0)

    left_line.best_fit = np.polyfit(plot_y, left_line.best_x, 2)
    right_line.best_fit = np.polyfit(plot_y, right_line.best_x, 2)

    # Calculate the radius of curvature of each line
    left_line.radius_of_curvature = get_line_curvature(left_line.best_x, plot_y, y_size - 1, x_scale=X_SCALE, y_scale=Y_SCALE)
    right_line.radius_of_curvature = get_line_curvature(right_line.best_x, plot_y, y_size - 1)

    left_bestx_fitted = ( left_line.best_fit[0] * plot_y ** 2 +
                          left_line.best_fit[1] * plot_y +
                          left_line.best_fit[2] )
    right_bestx_fitted = ( right_line.best_fit[0] * plot_y ** 2 +
                           right_line.best_fit[1] * plot_y +
                           right_line.best_fit[2] )
    left_line.line_base_pos = ((x_size // 2) - left_bestx_fitted[y_size - 1]) * X_SCALE
    right_line.line_base_pos = (right_bestx_fitted[y_size - 1] - (x_size // 2)) * X_SCALE

    if DEBUG_FIT_LINES:
        print(left_line.radius_of_curvature, 'm', right_line.radius_of_curvature, 'm')

    left_line.detected = True
    right_line.detected = True
    left_line.frames_not_detected = 0
    right_line.frames_not_detected = 0


def draw_overhead_lane(overhead_image, left_line, right_line):
    y_size = overhead_image.shape[0]
    plot_y = np.linspace(0, y_size - 1, y_size)

    overhead_zero = np.zeros_like(overhead_image).astype(np.uint8)
    overhead_color = np.dstack((overhead_zero, overhead_zero, overhead_zero))

    if not (left_line.best_x.size or right_line.best_x.size):
        return overhead_color
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.best_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.best_x,
                                                            plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    if left_line.frames_not_detected or right_line.frames_not_detected:
        cv2.fillPoly(overhead_color, np.int_([pts]), RED)
    else:
        cv2.fillPoly(overhead_color, np.int_([pts]), GREEN)

    return overhead_color


left_line = Line()
right_line = Line()
mtx = np.array([])
dist = np.array([])
def process_image(image):
    global left_line, right_line
    global mtx, dist

    if not mtx.size or not dist.size:
        if USE_SAVED_CORRECTION:
            mtx = SAVED_MTX
            dist = SAVED_DIST
        else:
            mtx, dist = calibrate_camera(CALIBRATION_DIR)

    # Undistort the original image
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    if DEBUG_PROCESS_IMAGE:
        plt.imshow(undist)
        plt.show()

    warp(undist, WARP_ORIG_SRC, WARP_OVERHEAD_DST)

    # Find pixels belonging to lane lines using color and gradient binary thresholding
    threshold_edges = image_threshold(undist)
    if DEBUG_PROCESS_IMAGE:
        plt.imshow(threshold_edges, cmap='gray')
        print('threshold edges')
        plt.show()

    # Mask detected pixels outside of region of interest
    masked_edges = region_of_interest(threshold_edges, MASK_ROI)
    if DEBUG_PROCESS_IMAGE:
        print('masked edges')
        plt.imshow(masked_edges, cmap='gray')
        plt.show()

    # Warp the image to give overhead perspective
    overhead_edges = warp(masked_edges, src=WARP_ORIG_SRC, dst=WARP_OVERHEAD_DST)
    if DEBUG_PROCESS_IMAGE:
        plt.imshow(overhead_edges, cmap='gray')
        plt.show()

    # Try to detect lane lines in the overhead image
    find_lane_lines(overhead_edges, left_line, right_line)

    # Draw the lane in a blank image
    overhead_lane = draw_overhead_lane(overhead_edges, left_line, right_line)

    # Warp the overhead blank back to original image space
    orig_perspective_lane = warp(overhead_lane, src=WARP_OVERHEAD_DST, dst=WARP_ORIG_SRC)

    # Combine the lane with the undistorted original image
    result = cv2.addWeighted(undist, 1, orig_perspective_lane, 0.3, 0)

    # Add radius and offset to image
    if left_line.best_fit.size and right_line.best_fit.size:
        curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2
        result = cv2.putText(result, 'Radius of lane curvature: {:.3f}m'.format(curvature),
                             (10, 35), cv2.FONT_HERSHEY_DUPLEX, 1, WHITE, 1, cv2.LINE_AA)

        offset = left_line.line_base_pos - right_line.line_base_pos
        if offset < 0:
            offset_dir = 'left'
        else:
            offset_dir = 'right'

        result = cv2.putText(result, 'Offset: {:.3f}m {} of center'.format(abs(offset), offset_dir),
                            (10, 75), cv2.FONT_HERSHEY_DUPLEX, 1, WHITE, 1, cv2.LINE_AA)

    if DEBUG_PROCESS_IMAGE:
        plt.imshow(result)
        plt.show()

    return result


def main():
#    np.set_printoptions(threshold=np.nan)
    if TEST_IMAGE:
        image = cv2.imread(TEST_IMAGE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process_image(image)
    else:
        clip1 = VideoFileClip(IN_VIDEO)
        out_clip = clip1.fl_image(process_image)
        out_clip.write_videofile(OUT_VIDEO, audio=False)

if __name__ == '__main__':
    main()
