import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os

CALIBRATION_DIR = './camera_cal'

DEBUG_IMAGE = False
DEBUG_FIT = False
USE_SAVED_CORRECTION = True


def calibrate_camera(calibration_dir, file_pattern='calibration*.jpg'):
    imgpoints = []
    objpoints = []

    objp = np.zeros((6 * 9, 3), np.float32)
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

#   1 - Simple Thresholding
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
#
#   2 - Adaptive Thresholding
#    grad_binary = cv2.adaptiveThreshold(scaled, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 4) 
#
#   3 - Otsu's Binarization
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

    if DEBUG_IMAGE:
        print('hls thresh')
        plt.imshow(channel, cmap='gray')
        plt.show()

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

    if DEBUG_IMAGE:
        print('rgb thresh')
        plt.imshow(channel, cmap='gray')
        plt.show()

    channel_binary = np.zeros_like(channel)
    channel_binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
#    channel_binary = cv2.adaptiveThreshold(channel, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5) 
    return channel_binary


def warp(image, inv=False, src=None, dst=None):
    if not src or dst:
        src = np.float32([[688, 448],
                          [1126, 720],
                          [188, 720],
                          [594, 448]])

        dst = np.float32([[960, 0],
                          [960, 720],
                          [320, 720],
                          [320, 0]])

#        src = np.float32([[585, 460],
#                          [203, 720],
#                          [1127, 720],
#                          [695, 460]])
#
#        dst = np.float32([[320, 0],
#                          [320, 720],
#                          [960, 720],
#                          [960, 0]])

    if not inv:
        m = cv2.getPerspectiveTransform(src, dst)
    else:
        m = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    if DEBUG_IMAGE:
        plt.imshow(image, cmap='gray')
        plt.plot((688, 1126), (448, 720), 'r')
        plt.plot((1126, 188), (720, 720), 'r')
        plt.plot((188, 594), (720, 448), 'r')
        plt.plot((594, 688), (448, 448), 'r')
        plt.show()
#        plt.imshow(image, cmap='gray')
#        plt.plot((585, 203), (460, 720), 'r')
#        plt.plot((203, 1127), (720, 720), 'r')
#        plt.plot((1127, 695), (720, 460), 'r')
#        plt.plot((695, 585), (460, 460), 'r')
#        plt.show()

        plt.imshow(warped, cmap='gray')
        plt.plot((960, 960), (0, 720), 'r')
        plt.plot((960, 320), (720, 720), 'r')
        plt.plot((320, 320), (720, 0), 'r')
        plt.plot((320, 960), (0, 0), 'r')
        plt.show()

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


def image_pipeline(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 180))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(80, 255))
    magnitude = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(60, 255))
    direction = dir_thresh(image, sobel_kernel=ksize, thresh=(0.9, 1.3))
    h = hls_thresh(image, 'h', thresh=(16, 30))
    s = hls_thresh(image, 's', thresh=(90, 255))
    r = rgb_thresh(image, 'r', (220, 255))
    g = rgb_thresh(image, 'g', (200, 255))

    if DEBUG_IMAGE:
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
    
    if DEBUG_IMAGE:
        plt.imshow(combined, cmap='gray')
        print('combined')
        plt.show()

    roi = np.array([[(650, 400), (1200, 720), (100, 720), (650, 400)]], dtype=np.int32)
    masked = region_of_interest(combined, roi)

    if DEBUG_IMAGE:
        print('masked')
        plt.imshow(masked, cmap='gray')
        plt.show()

    warped = warp(masked)
    return warped
   

def get_poly(overhead_image):
    histogram = np.sum(overhead_image[overhead_image.shape[0] // 2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((overhead_image, overhead_image, overhead_image)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[220:420]) + 220
    rightx_base = np.argmax(histogram[860:1060]) + 860
    
    # Choose the number of sliding windows
    nwindows = 15 
    # Set height of windows
    window_height = np.int(overhead_image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = overhead_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 200
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = overhead_image.shape[0] - (window + 1) * window_height
        win_y_high = overhead_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
#        print("Left Inds:", len(good_left_inds))
        if len(good_left_inds) > minpix:
#            print("Window:", window)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, overhead_image.shape[0] - 1, overhead_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if DEBUG_FIT: 
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 600 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
#    if DEBUG_FIT:
#    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    return (ploty, left_fitx, right_fitx)


def process_image(image):
    mtx = np.array([])
    dist = np.array([])
    if USE_SAVED_CORRECTION:
        mtx = np.array([[ 1.15158804e+03, 0.00000000e+00, 6.66167057e+02],
                        [ 0.00000000e+00, 1.14506859e+03, 3.86440204e+02],
                        [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.array([[ -2.35510339e-01, -7.90388401e-02, -1.28492202e-03,
                           8.25970342e-05, 7.22743174e-02]])
    if not mtx.size or not dist.size:
        mtx, dist = calibrate_camera(CALIBRATION_DIR)

    undist = cv2.undistort(image, mtx, dist, None, mtx)

    if DEBUG_FIT:
        plt.imshow(undist)
        plt.show()

    w = warp(undist)
    if DEBUG_FIT:
        plt.imshow(w)
        plt.show()

 
    overhead = image_pipeline(undist)
    if DEBUG_FIT:
        plt.imshow(overhead, cmap='gray')
        plt.show()

    ploty, left_fitx, right_fitx = get_poly(overhead)
    warp_zero = np.zeros_like(overhead).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space
    newwarp = warp(color_warp, inv=True)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if DEBUG_FIT:
        plt.imshow(result)
        plt.show()
    return result


from moviepy.editor import VideoFileClip
def main():
    np.set_printoptions(threshold=np.nan)

    image = cv2.imread('./test_images/straight_lines1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    process_image(image)

    clip_out = './processed_video.mp4'
    clip1 = VideoFileClip("./project_video.mp4")#.subclip(0, 5)
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(clip_out, audio=False)

if __name__ == '__main__':
    main()
