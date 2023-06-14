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

# How to use when converting a YOLOv7 dataset
# python apply-fisheye.py --input_dir=VisDrone/VisDrone2019-DET-test-dev -f=<choose an input focal length>
#   T

# Adjusting the current fisheye distortion function to 
def normalised_fisheye_distort(input_image, f, h, w, u_x, u_y):
    # Create template output image array based on the input image shape
    result = np.zeros_like(input_image)

    # Might need special conditions for the middle axes

    # Iterate along the top right quadrant only; top right since this 
    #   avoids having to deal with negative values
    for y in range(u_y): # Note that we iterate through the raw coordinate values
        for x in range(u_x,w):
            # Normalise to be between the ranges 0 to 1 and relative to the centre (0.5)
            # norm_y = y-u_y
            # norm_x = x-u_x

            # print("Original coordinate : (", x-u_x, ", ", y-u_y, ")")
            norm_y = (y-u_y)/(2*u_y)
            norm_x = (x-u_x)/(2*u_x)
            # print("Normalised coordinate : (", norm_x, ", ", norm_y, ")")

            # Get gradient of line going through x,y
            m = norm_y if norm_x==0 else (norm_y)/(norm_x)
            # print("m: ", m)
            rc = math.sqrt(pow(norm_x,2) + pow(norm_y,2))
            # print("rc: ", rc)
            numerator = pow((f*(math.atan(rc/f))),2)
            denominator = 1+pow(m,2)
            x_f = math.sqrt(numerator/denominator)
            y_f = x_f * m

            # print("Newly calculated coordinate : (", x_f, ", ", y_f, ")")

            # Scale back to original values
            x_f = x_f * u_x
            y_f = y_f * u_y
            # print("Rescaled distorted coordinate : (", x_f, ", ", y_f, ")")

            x_f = int(x_f)
            y_f = int(y_f)

            # Make use of the maths for all 4 quadrants
            # print("Final relative coordinate : (", x_f, ", ", y_f, ")")
            # print("Destination coordinate : (", x_f+u_x, ", ", y_f+u_y, ")")
            result[y_f+u_y, x_f+u_x] = input_image[y, x] # TR
            result[y_f+u_y, w-(x_f+u_x+1)] = input_image[y, w-(x+1)] # TL
            result[h-(y_f+u_y+1), w-(x_f+u_x+1)] = input_image[h-(y+1), w-(x+1)] # BL
            result[h-(y_f+u_y+1), x_f+u_x] = input_image[h-(y+1), x] # BR
            # time.sleep(1)
            # print(" ")
    return result

# Optimised fisheye distortion function that only makes calculates on one
#   quadrant of the input coordinates and uses this to map the changes in
#   the other 3 quadrants
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

    # Might need special conditions for the middle axes

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

# Function to go through the associated annotation data file and annotate as such
def apply_annotations(input_image, annotation_file):
    # input_image - image we are applying the boundary box annotations to
    # annotation_data - file containing the needed information to make 
    #   the boundary boxes

    annotation_data = open(annotation_file)
    for line in annotation_data: # Iterate through each line of the annotation file
        # bbox_l = left-most x coordinate (smallest val)
        # bbox_t = top-most y coordinate (smallest val)
        # bbox_w = width of the boundary box
        # bbox_h = height of the boundary box
        # score - 1 indicates that the bounding box is considered, 0 means it is ignored
        # cat - class of the object
        bbox_l, bbox_t, bbox_w, bbox_h, score, cat, _, _ = map(int, line.split(","))

        if score == 0 or cat == 0 or cat == 11:
            continue # ignore these boundary boxes

        top_left = (bbox_l, bbox_t)
        bottom_right = (bbox_l+bbox_w, bbox_t+bbox_h)

        # Plot a thin red boundary box based on the given coordinates
        input_image = cv2.rectangle(input_image, top_left, bottom_right, (0,0,255), 1)
    return input_image

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

# Function to apply the same mathematics as the distortion equation to the
#   boundary box corner coordinates and write it to a separate file
def distort_annotations(annotation_file, f, u_x, u_y, crop=False, top_border=0, left_border=0):
    # Create a new file, return an error if it already exists
    original_file = open(annotation_file)
    parent_dir = annotation_file.parents[1]
    # print(str(annotation_file.parent).split("\\")[1])
    result_file = open(str(parent_dir) + "/fisheye/" + args.focal_length + "/annotations/" + annotation_file.stem+".txt", "w")

    for line in original_file:
        bbox_l, bbox_t, bbox_w, bbox_h, score, cat, trunc, occl = map(int, line.split(","))

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

        ret_items = [bbox_l, bbox_t, bbox_w, bbox_h, score, cat, trunc, occl]

        result_file.write(",".join([str(n) for n in ret_items])+"\n")
    # Return the filepath that the annotations have been written to
    return result_file.name

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

    # norm_y = y/h - 0.5
    # norm_x = x/w - 0.5

    # Get gradient of line going through x,y
    m = norm_y if norm_x==0 else (norm_y)/(norm_x)
            
    # rf = f.arctan(rc/f)
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

# def upscale_image(input_img, target_res):
#     return cv2.resize(input_img, dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC)

def main():
    if not args.focal_length:
        print("Focal length not given")

    if args.input:
        input_image = cv2.imread("example_data/images/"+args.input+".jpg")
        # input_image_cpy = copy.deepcopy(input_image)
        # cv2.imshow("Original image", input_image)

        # file = open("example_data/annotations/"+args.input+".txt")
        # data = file.readlines()
        # original_bb = apply_annotations(input_image_cpy, data)
        # cv2.imshow("Original image with boundary boxes", original_bb)

        # Get input image resolution for fisheye calculations
        h, w, _ = input_image.shape
        u_x = int(w/2)
        u_y = int(h/2)
        if w/2 % 2 == 0:
            u_x -= 1
        if h/2 % 2 == 0:
            u_y -= 1

        # Time this function
        start = time.time()
        distorted_img = optim_fisheye_distort(input_image, int(args.focal_length), h, w, u_x, u_y)
        end = time.time()
        print("Time taken = ", end-start)
        # Time taken for the example VisDrone image
        # a) 1.3572
        # b) 0.6970 - after repeating maths along different quadrants

        # distorted_img_cpy = copy.deepcopy(distorted_img)
        cv2.imshow("Distorted image", distorted_img)
        # distort_annotations(data, args.input, f, u_x, u_y)
        # file = open("example_data/distorted/annotations/"+args.input+".txt")
        # data = file.readlines()
        # distorted_bb = apply_annotations(distorted_img_cpy, data)
        # cv2.imshow("Distorted image with boundary boxes", distorted_bb)
        # result_img = fisheye_distort_2(input_image, 1.1)
        cv2.waitKey()
    elif args.input_dir:
        # Expected input directory is the parent directory for the images
        #   in the subset of that dataset
        input_images_dir = Path(args.input_dir+"/images/")
        # input_annotations_dir = Path(args.input_dir+"/annotations/")
        input_labels_dir = Path(args.input_dir+"/labels/")

        result_images_dir = Path(args.input_dir+"/fisheye/" + args.focal_length + "/images/")
        # result_annotations_dir = Path(args.input_dir+"/fisheye/" + args.focal_length + "/annotations/")
        result_labels_dir = Path(args.input_dir+"/fisheye/" + args.focal_length + "/labels/")
        result_images_dir.mkdir(parents=True, exist_ok=True)
        # result_annotations_dir.mkdir(parents=True, exist_ok=True)
        result_labels_dir.mkdir(parents=True, exist_ok=True)

        # upscaled_dir = Path(args.input_dir+"/upscaled/")
        # upscaled_results_dir = Path(args.input_dir+"/upscaled/results/" + args.focal_length + "/")

        # upscaled_dir.mkdir(parents=True, exist_ok=True)
        # upscaled_results_dir.mkdir(parents=True, exist_ok=True)

        input_images = sorted(input_images_dir.glob('*.jpg'))
        # input_annotations = sorted(input_annotations_dir.glob('*.txt'))
        input_labels = sorted(input_labels_dir.glob('*.txt'))

        for i,v in enumerate(input_images):
            # print(dir_input.stem +  i.stem + "_test.jpg")
            input_image = cv2.imread(str(v))

            # ======= Upscale the image based on the current resolutions =======
            h, w, _ = input_image.shape
            scale = 3840/w
            # 1920, 1080
            # upscaled_image = cv2.resize(input_image, dsize=(1920, 1080), interpolation=cv2.INTER_CUBIC)
            # upscaled_image = upscaled_image_changed.copy()
            upscaled_image = cv2.resize(input_image, dsize=(3840, int(h*scale)), interpolation=cv2.INTER_CUBIC) # 3840 x 2160 (19x6)
            # upscaled_image_changed = upscaled_image.copy()

            # upscaled_image_2 = cv2.resize(input_image, dsize=(1280, 960), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("Original image ", input_image)
            # cv2.imshow("Upscaled image: ", upscaled_image)

            # Image labelling (for display reasons)
            # labeled_img = apply_labels(upscaled_image_changed, input_labels[i])
            # cv2.imshow("Upscaled image with boundary boxes", labeled_img)

            # cv2.imwrite(str(upscaled_dir) + "/" + v.stem + "_1920x1080.jpg", upscaled_image)
            # cv2.imwrite(str(upscaled_dir) + "/" + v.stem + "_3840x2160.jpg", upscaled_image_2)
            # cv2.imwrite(str(upscaled_dir) + "/" + v.stem + "_1280x960.jpg", upscaled_image_2)
            # cv2.waitKey()

            # Get dimensions
            h, w, _ = upscaled_image.shape
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

            # top_border = dist_tm[1]
            # left_border = dist_ml[0]

            # Apply a fisheye distortion to the upscaled image
            # result_img = optim_fisheye_distort(input_image, int(args.focal_length), h, w, u_x, u_y)
            result_img = optim_fisheye_distort(upscaled_image, int(args.focal_length), h, w, u_x, u_y)
            # cv2.imshow("Original", result_img)
            # Crop the distorted image
            cropped_result_img = result_img[top_border:(h-top_border), left_border:w-left_border]
            # cv2.imshow("Cropped result image", cropped_result_img)
            cv2.imwrite(str(result_images_dir) + "/" + v.stem + ".jpg", cropped_result_img)

            # upscaled_cropped_image = cv2.resize(cropped_result_img, dsize=(1280,960), interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite(str(upscaled_results_dir) + "/" + v.stem + "_1280x960_upscaled.jpg", upscaled_cropped_image)
            # cv2.imshow("Upscaled cropped image", upscaled_cropped_image)
            # cv2.waitKey()

            # Save the 
            # cv2.imwrite(str(upscaled_results_dir) + "/" + v.stem + "_1920x1080.jpg", cropped_result_img)
            # cv2.imwrite(str(upscaled_results_dir) + "/" + v.stem + "_3840x2160.jpg", cropped_result_img)
            # cv2.imwrite(str(upscaled_results_dir) + "/" + v.stem + "_1280x960.jpg", cropped_result_img)
            # cv2.imshow("Original image resolution with fisheye", result_img)
            # cv2.waitKey()

            # Distort the labels and save (within the function)
            dist_label_file = distort_labels(input_labels[i], int(args.focal_length), u_x, u_y, 
                                                crop=True, top_border=top_border, left_border=left_border)

            # distorted_bb = apply_labels(cropped_result_img, dist_label_file)
            # cv2.imwrite("test.jpg", distorted_bb)
            # cv2.imshow("Distorted image with boundary boxes", distorted_bb)
            # cv2.waitKey()
    else: # No input image or path added, default to standard code
        print("No input image or directory given")

if __name__ == "__main__":
    main()