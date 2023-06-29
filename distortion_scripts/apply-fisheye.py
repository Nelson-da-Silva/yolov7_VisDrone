# How to use when converting a YOLOv7 dataset
# python distortion_scripts/apply-fisheye.py --input_dir=VisDrone/VisDrone2019-DET-test-dev -f=<choose an input focal length>
#   - where the input directory has two subdirectories: images and labels
#   files are given in the YOLOv7 notation

import cv2
import numpy as np
import math
import argparse
import time
import copy

from pathlib import Path

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i')
parser.add_argument('--input_dir', '-d')
parser.add_argument('--focal_length', '-f')
args = parser.parse_args()

# Optimised fisheye distortion function that only makes calculates on one
#   quadrant of the input coordinates and uses this to map the changes in
#   the other 3 quadrants
# TODO: Currently an issue with the distortion model where distortions appear different at
#   the same focal length if the input resolution is different. Can fix this by having all
#   images be reshaped/upscaled to a specific resolution prior to applying the distortion. 
#   Need find a resolution for all images to be reshaped to such that the distortion parameter
#   f, actually applies distortion in a way that the focal length affects distortion. May need
#   look back at the maths behind the model.
def optim_fisheye_distort(input_image, f, h, w, u_x, u_y):
    """""
        input_image = image to be distorted
        f = focal length
        h = height of the input image
        w = width of the input image
        u_x = principal point (x-coordinate)
        u_y = principal point (y-coordinate)
    """""

    # Create template output image array based on the input image shape
    result = np.zeros_like(input_image)

    # Iterate along the top right quadrant only; top right since this 
    #   avoids having to deal with negative values
    for y in range(u_y):
        for x in range(u_x,w):
            norm_y = y-u_y
            norm_x = x-u_x

            # Get gradient of line going through x,y
            m = norm_y if norm_x==0 else (norm_y)/(norm_x)
            rc = math.sqrt(pow(norm_x,2) + pow(norm_y,2))
            numerator = pow((f*(math.atan(rc/f))),2)
            denominator = 1+pow(m,2)
            x_f = math.sqrt(numerator/denominator)
            y_f = x_f * m

            x_f = int(x_f)
            y_f = int(y_f)

            # Make use of the maths for all 4 quadrants
            result[y_f+u_y, x_f+u_x] = input_image[y, x] # TR
            result[y_f+u_y, w-(x_f+u_x+1)] = input_image[y, w-(x+1)] # TL
            result[h-(y_f+u_y+1), w-(x_f+u_x+1)] = input_image[h-(y+1), w-(x+1)] # BL
            result[h-(y_f+u_y+1), x_f+u_x] = input_image[h-(y+1), x] # BR
    return result

# Non-optimised distortion function, iterates through all quadrants
def fisheye_distort(input_image, f, h, w, u_x, u_y):
    """""
        input_image = image to be distorted
        f = focal length
        h = height of the input image
        w = width of the input image
        u_x = principal point (x-coordinate)
        u_y = principal point (y-coordinate)
    """""

    # Create template output image array based on the input image shape
    result = np.zeros_like(input_image)

    # Iterate through the input image coordinates, calculate their new positions 
    #   and fill that value as such in the result template
    for y in range(h): # Iterate vertically
        for x in range(w): # Iterate horizontally
            # Can we reuse the maths? TODO optimise to reuse the coordinate measurements throughout
            ret_x, ret_y = distort_coordinates(x,y,u_x,u_y,f)
            result[ret_y][ret_x] = input_image[y][x]
    return result

# Function to distort the label file associated with the image being distorted such that
#   the labels are distorted in the same way. Note that the bounding boxes will not exactly match
#   the distorted objects' shapes now due to how distortion is computed on the corner points which
#   may not exactly match the points on the object therefore the distorted labels won't be perfect
#   but slightly off.
def distort_labels(label_file, f, u_x, u_y, crop=False, top_border=0, left_border=0):
    # Create a new file, return an error if it already exists
    original_file = open(label_file)
    parent_dir = label_file.parents[1]
    # print(str(label_file.parent).split("\\")[1])
    result_file = open(str(parent_dir) + "/fisheye/" + args.focal_length + "/labels/" + label_file.stem+".txt", "w")

    u_x_crop = u_x
    u_y_crop = u_y

    if crop:
        u_x_crop = u_x - left_border
        u_y_crop = u_y - top_border

    for line in original_file:
        inst_class, x_centre, y_centre, bbox_w, bbox_h = map(float, line.split(" "))
        x_centre = x_centre * u_x*2
        bbox_w = int(bbox_w * u_x*2)
        y_centre = y_centre * u_y*2
        bbox_h = int(bbox_h * u_y*2)

        bbox_l = int(x_centre - bbox_w/2)
        bbox_t = int(y_centre - bbox_h/2)

        dist_tl = distort_coordinates(bbox_l, bbox_t, u_x, u_y, f)
        dist_bl = distort_coordinates(bbox_l, bbox_t+bbox_h, u_x, u_y, f)
        dist_br = distort_coordinates(bbox_l+bbox_w, bbox_t+bbox_h, u_x, u_y, f)
        dist_tr = distort_coordinates(bbox_l+bbox_w, bbox_t, u_x, u_y, f)

        # Calculate new metrics from the corner distorted corner coordinates
        bbox_l = min(dist_tl[0], dist_bl[0]) # Left-most x coordinate
        bbox_t = min(dist_tl[1], dist_tr[1]) # Top-most y coordinate
        bbox_w = max(dist_br[0], dist_tr[0]) - bbox_l
        bbox_h = max(dist_br[1], dist_bl[1]) - bbox_t

        if crop: # Determine 
            bbox_l -= left_border
            bbox_t -= top_border

        # Convert back to format for label file
        x_centre = (bbox_l + bbox_w/2)/(u_x_crop*2)
        y_centre = (bbox_t + bbox_h/2)/(u_y_crop*2)

        ret_items = [inst_class, x_centre, y_centre, bbox_w/(u_x_crop*2), bbox_h/(u_y_crop*2)]

        result_file.write(" ".join([str(n) for n in ret_items])+"\n")
    # Return the filepath that the annotations have been written to
    return result_file.name

def distort_coordinates(x, y, u_x, u_y, f):
    norm_y = y-u_y
    norm_x = x-u_x

    # Get gradient of line going through x,y
    m = norm_y if norm_x==0 else (norm_y)/(norm_x)
            
    # Formula: rf = f.arctan(rc/f)
    # Calculate hypotenuse length of pixel from conventional image
    rc = math.sqrt(pow(norm_x,2) + pow(norm_y,2))
    # print("Calculated rc: ", rc)
            
    numerator = pow((f*(math.atan(rc/f))),2)
    # print("Calculated numerator: ", numerator)
    denominator = 1+pow(m,2)
    # print("Calculated denominator: ", denominator)
    x_f = math.sqrt(numerator/denominator)

    if norm_x < 0: # If x is originally negative, make x_f negative
        x_f = -x_f
    # print("Calculated x_f: ", x_f)
    y_f = x_f * m
    # print("Calculated y_f: ", y_f)

    # Resulting points are relative to u_x and u_y
    # print("Mapping : ", x, ", ", y, " : ", x_f+u_x, ", ", y_f+u_y)

    # Return the distorted pair of coordinates
    return int(x_f + u_x), int(y_f + u_y)

# Helper functions
def apply_labels(input_image, label_file):
    label_data = open(label_file)
    h, w, _ = input_image.shape
    for line in label_data: # Iterate through each line of the annotation file
        inst_class, x_centre, y_centre, bbox_w, bbox_h = map(float, line.split(" "))
        # Box coordinates must be in normalised xywh format (from 0 to 1)
        x_centre = x_centre * w
        bbox_w = int(bbox_w * w)
        y_centre = y_centre * h
        bbox_h = int(bbox_h * h)

        bbox_l = int(x_centre - bbox_w/2)
        bbox_t = int(y_centre - bbox_h/2)

        top_left = (bbox_l, bbox_t)
        bottom_right = (bbox_l+bbox_w, bbox_t+bbox_h)
        # Plot a thin red boundary box based on the given coordinates
        input_image = cv2.rectangle(input_image, top_left, bottom_right, (0,0,255), 1)
    return input_image

# Radial distortion model function added for completeness - Not in use
# https://ieeexplore.ieee.org/abstract/document/9066935/references#references
def fisheye_distort_2(input_image, r):
    # input_image = image to be distorted
    # f = focal length

    # Create template output image array based on the input image shape
    result = np.zeros_like(input_image)
    # result = np.full(input_image.shape, 150, np.uint8)

    # Calculate principal points (just being taken as the centre of the image)
    h, w, _ = input_image.shape
    print(input_image.shape)

    u_x = int(w/2)
    u_y = int(h/2)
    if w/2 % 2 == 0:
        u_x -= 1
    if h/2 % 2 == 0:
        u_y -= 1

    # u_x = int(w/2) - 1
    # print("u_x : ", u_x)
    # u_y = int(h/2)
    # print("u_y : ", u_y)

    # Iterate through the input image coordinates, calculate their new positions and fill that value as such in the result template
    for y in range(h): # Iterate vertically
        # y = h - 1 - y
        for x in range(w): # Iterate horizontally
            # Form normalised coordinates
            y_n = (y-u_y)/u_y
            x_n = (x-u_x)/u_x

            x_f = x_n*math.sqrt(1-(y_n**2)/2) * u_x
            y_f = y_n*math.sqrt(1-(x_n**2)/2) * u_y

            # Coordinate scaling factor to contorl severity of distortion
            x_f = x_f*math.e**(-r**2)
            y_f = y_f*math.e**(-r**2)

            # print("Calculated x_f: ", x_f)
            # print("Calculated y_f: ", y_f)

            # Resulting points are relative to u_x and u_y
            # print("Mapping : ", x, ", ", y, " : ", x_f+u_x, ", ", y_f+u_y)

            result[int(y_f + u_y)][int(x_f + u_x)] = input_image[y][x]
    return result

def main():
    if not args.focal_length:
        print("Focal length not given")

    if args.input_dir:
        # Expected input directory is the parent directory for the images
        input_images_dir = Path(args.input_dir+"/images/")
        input_labels_dir = Path(args.input_dir+"/labels/")

        # Make directory to store results
        result_images_dir = Path(args.input_dir+"/fisheye/" + args.focal_length + "/images/")
        result_labels_dir = Path(args.input_dir+"/fisheye/" + args.focal_length + "/labels/")
        result_images_dir.mkdir(parents=True, exist_ok=True)
        result_labels_dir.mkdir(parents=True, exist_ok=True)

        input_images = sorted(input_images_dir.glob('*.jpg'))
        input_labels = sorted(input_labels_dir.glob('*.txt'))

        for i,v in enumerate(input_images):
            input_image = cv2.imread(str(v))

            # ======= Image labelling (for display reasons) =======
            # Uncomment to apply visualise the upscaled image prior to distortions
            input_image_changed = input_image.copy()
            labeled_img = apply_labels(input_image_changed, input_labels[i])
            cv2.imshow("Normal image with boundary boxes", labeled_img)

            # ======= Distort the image =======
            # Get dimensions
            h, w, _ = input_image.shape
            u_x = int(w/2)
            u_y = int(h/2)
            if w/2 % 2 == 0:
                u_x -= 1 
            if h/2 % 2 == 0:
                u_y -= 1

            # Calculate boundary coordinates of the distorted image by looking 
            #   at the top middle and middle left coordinates; used for cropping the image
            #   and distorting the annotation coordinates.
            _, top_border = distort_coordinates(u_x, 0, u_x, u_y, int(args.focal_length)) # top_middle = (u_x, 0), dist_tm
            left_border, _ = distort_coordinates(0, u_y, u_x, u_y, int(args.focal_length)) # middle_left = (0, u_y), dist_ml

            # Apply a fisheye distortion to the upscaled image
            result_img = optim_fisheye_distort(input_image, int(args.focal_length), h, w, u_x, u_y)
            # cv2.imshow("Original", result_img)
            # Crop the distorted image
            cropped_result_img = result_img[top_border:(h-top_border), left_border:w-left_border]
            # cv2.imshow("Cropped result image", cropped_result_img)
            cv2.imwrite(str(result_images_dir) + "/" + v.stem + ".jpg", cropped_result_img)

            # Distort the labels and save (within the function)
            dist_label_file = distort_labels(input_labels[i], int(args.focal_length), u_x, u_y, 
                                                crop=True, top_border=top_border, left_border=left_border)

            # ======= Visualise the distorted image with it's newly mapped bounding boxes =======
            # distorted_bb = apply_labels(cropped_result_img, dist_label_file)
            # cv2.imwrite("test.jpg", distorted_bb)
            # cv2.imshow("Distorted image with boundary boxes", distorted_bb)
            # cv2.waitKey()
    else: # No input image or path added, default to standard code
        print("No input image or directory given")

if __name__ == "__main__":
    main()