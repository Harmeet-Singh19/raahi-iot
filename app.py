from email.mime import image
from types import coroutine
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pickle
import os
import jsonpickle
import numpy as np
import cv2
import requests
import urllib.request
import base64

import glob

from random import randint

# Un-updated app 

def fit_poly(img_shape, leftx, lefty, rightx, righty):
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	return left_fitx, right_fitx, ploty



# build app
app = Flask(__name__)
CORS(app)
# os.makedirs('temporary',exist_ok = True)
# video_path =  "video.mp4"
# model = predict.load_model('./models/model.h5')
image_path = "image.png"

@app.route('/save', methods=['GET'])
def user_download():
    
    url = request.values.get('url')
    print(url)
    alt = request.values.get('token')
    print(alt)
    r = requests.get(url, allow_redirects = True)
    url += "&token="
    url += alt

    print(url)
    with open(image_path, 'wb') as img:
        img.write(r.content)
    img.close()  

    final_prediction = {}


    image_ext = cv2.imread(image_path)
    initial_image = np.copy(image_ext)
    
    
    crop_img = image_ext[0:0+400, 0:0+600]
    bigger = cv2.resize(crop_img, (1280, 720))

    image_ext = bigger
    initial_image= np.copy(image_ext)
    
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images_for_calibration = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for f_name in images_for_calibration:
            img_read = cv2.imread(f_name)
            gray = cv2.cvtColor(img_read,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, initial_image.shape[1::-1], None, None)
    undistorted = cv2.undistort(initial_image, mtx, dist, None, mtx)
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    src = np.float32([[590,450],[687,450],[1100,720],[200,720]])
    dst = np.float32([[300,0],[900,0],[900,720],[300,720]])
    
    im_size = (combined_binary.shape[1], combined_binary.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)

    warped_image = cv2.warpPerspective(combined_binary, M, im_size, flags=cv2.INTER_NEAREST)
    left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
    right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])
    
    margin = 100
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])



    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    left_fitx, right_fitx, ploty = fit_poly(warped_image.shape, leftx, lefty, rightx, righty)
    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, M_inverse, im_size)
    
    result_final = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    
    output_image_after_detecting = result_final
    
    dir = ""
    if leftx[-1] - leftx[0] > rightx[-1] - rightx[0]:
        dir += "left"
    elif leftx[-1] - leftx[0] < rightx[-1] - rightx[0]:
        dir += "right"
    else:
        dir += "straight"

    final_prediction['result'] = dir

    # os.remove(video_path)
    return jsonify(final_prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
