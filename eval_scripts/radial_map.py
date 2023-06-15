# Input predictions text file must include confidences

import cv2
import numpy as np
import math
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i')
# parser.add_argument('--input_dir', '-d')
parser.add_argument('--input_images', '-img')
parser.add_argument('--input_labels', '-lbl')
parser.add_argument('--input_preds', '-prd')
parser.add_argument('--focal_length', '-f')
args = parser.parse_args()

def apply_tensor_labels(input_image, label_tensor, colour):
    h, w, _ = input_image.shape
    label_tensor[:,1:] = xyxy2xywh(label_tensor[:,1:])
    for i,line in enumerate(label_tensor): # Iterate through each line of the annotation file
        # print(line)
        # print(len(line))
        if len(line) == 6: # conf
            inst_class, x_centre, y_centre, bbox_w, bbox_h, _ = line
        else:
            inst_class, x_centre, y_centre, bbox_w, bbox_h = line
            # print(x_centre)
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
        input_image = cv2.circle(input_image, (int(x_centre), int(y_centre)), 0, [0,255,0], 1)
        input_image = cv2.rectangle(input_image, top_left, bottom_right, colour, 1)
    return input_image

# Function taken from test.py
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def create_tensor_from_file(input_file):
    input_data = open(input_file)
    first_line = input_data.readline()
    n = len(first_line.split(" "))
    if not first_line:
        print("No labels in this image, skip to next image")
        return
    output_tensor = torch.FloatTensor([float(i) for i in first_line.split(" ")])
    output_tensor = torch.reshape(output_tensor, [1,n])
    for v in input_data:
        output_tensor = torch.vstack((output_tensor, torch.FloatTensor([float(i) for i in v.split(" ")])))
    # Convert dimensions to xyxy format
    output_tensor[:,1:5] = xywh2xyxy(output_tensor[:,1:5])
    return output_tensor

def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    # if plot:
        # plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        # plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def round_up(x, a):
    # print("x/a ", (x/a))
    return math.ceil(x/a)*a

# Input dimensions are of the heatmap/range we want to discretize between.
# Returns int values (so range must be larger than 1).
def discretize_dimensions(input_tensor_line, x_dim, y_dim):
    # Get dimensions of the label
    x_left = round(round_up(float(input_tensor_line[1]),1/x_dim)*x_dim) - 1
    if x_left == -1:
        x_left = 0
    # print(labels_tensor[d][0][1], " mapped to ", x_left)
    x_right = 0
    if math.floor(float(input_tensor_line[3])) == 1:
        y_right = y_dim-1
    else:
        x_right = round(round_up(float(input_tensor_line[3]),1/x_dim)*x_dim) - 1
    # print(labels_tensor[d][0][3], " mapped to ", x_right)
    # Appending to the heatmap should be reversed vertically
    #   since (0,0) is at the top left
    y_left = round(round_up(input_tensor_line[2],1/y_dim)*y_dim) - 1
    if y_left == -1:
        y_left = 0
    # print(labels_tensor[d][0][2], " mapped to ", y_left)
    y_right = 0
    if math.floor(float(input_tensor_line[4])) == 1:
        y_right = y_dim-1
    else:
        y_right = round(round_up(input_tensor_line[4],1/y_dim)*y_dim) - 1
    return x_left, x_right, y_left, y_right

# Take in an input tensor of predictions and a range of rectangular regions to cover, output
#   a filtered version of that tensor to only include the labels in those regions
def filter_tensor(input_tensor, n_regions):
    filtered_indexes = [[] for _ in range(n_regions)] # Setting up empty lists to store the indexes
    #   of the predictions belonging to the n-1th region
    # iou_hm = np.zeros((n_regions*2-1,n_regions*2-1)) # Test heatmap
    # print(iou_hm)

    # Iterate through the predictions row by row
    for i, row in enumerate(input_tensor):
        # print("Row: ", row)
        # Calculate discretized dimensions ranges, input dimensions depends on the number of regions we want to evaluate
        x_left, x_right, y_left, y_right = discretize_dimensions(row, n_regions*2-1, n_regions*2-1)
        # print("Discretized dimensions: ", discretize_dimensions(row, n_regions*2-1, n_regions*2-1))

        # # Graph plotting for testing
        # for l in range(x_left,x_right+1):
        #     for k in range (y_left, y_right+1):
        #         iou_hm[k,l] += 1
        # continue

        # Classify what region that should go into by iterating through all boxes from that object
        #   and seeing what region it belongs to
        regions = set() # Regions the current prediction is already in
        for l in range(x_left,x_right+1):
            for k in range (y_left, y_right+1):
                # print("Index : (", l, ", ", k, ")")
                l_corrected = abs(l-(n_regions-1))
                r = 0
                if abs(k-(n_regions-1)) <= l_corrected:
                    r = l_corrected+1
                else:
                    r = abs(k-(n_regions-1))+1
                # print("Placed in region: ", r)

                # Check what regions they coincide with and append their index to that region if so
                if r not in regions: # Region not yet appended
                    regions.add(r)
                    filtered_indexes[r-1].append(i) # Append the current index to the indices of that region
                else:
                    continue # skip
    # sns.heatmap(iou_hm, cmap='crest')
    # plt.show()

    # Return a list of filtered_tensors of size n_regions, tensor for each region
    return filtered_indexes

# Helper function to plot the objects in an image in a heatmap style to test filtering
#   highlight - region we want to highlight in the graph
def plot_boxes_map(input_tensor, n_regions, all_labels, highlight=0):
    if highlight != 0: # Highlight a specific region
        hl_map = np.zeros((n_regions*2-1,n_regions*2-1))
        for l in range(-(highlight-1), highlight):
            print(l)
            hl_map[highlight-1 + (n_regions-1),l+(n_regions-1)] += 1
            hl_map[-(highlight-1)+ (n_regions-1),l+(n_regions-1)] += 1
        for k in range(-(highlight-1), highlight):
            hl_map[k+(n_regions-1), highlight-1 + (n_regions-1)] += 1
            hl_map[k+(n_regions-1), -(highlight-1) + (n_regions-1)] += 1
        plt.subplot(2,2,1)
        plt.title("Region in focus")
        sns.heatmap(hl_map, cmap='crest')
        # plt.show()

    labels_map = np.zeros((n_regions*2-1,n_regions*2-1)) # Test heatmap
    for row in input_tensor:
        x_left, x_right, y_left, y_right = discretize_dimensions(row, n_regions*2-1, n_regions*2-1)
        for l in range(x_left,x_right+1):
            for k in range (y_left, y_right+1):
                labels_map[k,l] += 1

    all_labels_map = np.zeros((n_regions*2-1,n_regions*2-1)) # Test heatmap
    for row in all_labels:
        x_left, x_right, y_left, y_right = discretize_dimensions(row, n_regions*2-1, n_regions*2-1)
        for l in range(x_left,x_right+1):
            for k in range (y_left, y_right+1):
                all_labels_map[k,l] += 1

    plt.subplot(2,2,2)
    plt.title("Labels within that region")
    sns.heatmap(labels_map, cmap='crest')

    plt.subplot(2,2,3)
    plt.title("All labels")
    sns.heatmap(all_labels_map, cmap='crest')

    plt.show()

# Filter out labels in circular regions of the image
# Inputs
#   - input_tensor - tensor of boundary boxes with coordinates to be filtered out
#   - img_shape - resolution of the input image we are filtering into regions for
def circular_filter(input_tensor, img_shape):
    copy_tensor = input_tensor.detach().clone()
    copy_tensor[:,1:5] = xyxy2xywh(copy_tensor[:,1:5]) # Convert to XYWH format
    filtered_indexes = [[] for _ in range(3)] # Setting up empty lists to store the indexes
    scale = img_shape[1]/img_shape[0] # e.g. 16/9, to multiply the x-axis by

    for index, row in enumerate(copy_tensor):
        if len(row) == 6: # conf format
            _, x_centre, y_centre, _, _, _ = row
        else:
            _, x_centre, y_centre, _, _ = row

        # Scale coordinates and centre at the middle
        x_centre -= 1/2
        x_centre *= scale
        y_centre -= 1/2

        # Calculate radius of boundary box centre from the point ine middle of the image
        r = math.sqrt(x_centre**2 + y_centre**2)
        if r <= scale*1/6: # Region 1
            filtered_indexes[0].append(index)
        elif r > scale*1/6 and r <= scale*0.4: # Region 2
            filtered_indexes[1].append(index)
        elif r > scale*0.4: # Region 3
            filtered_indexes[2].append(index)

    return filtered_indexes

def main():
    if not args.input_images and not args.input_labels and not args.input_preds:
        print("No input directory provided")
    else:
        # input_images_dir = Path(args.input_dir+"/images/")
        # input_labels_dir = Path(args.input_dir+"/labels/")
        # input_preds_dir = Path(args.input_dir+"/preds/")
        # input_og_preds_dir = Path(args.input_dir+"/og_preds/")

        input_images_dir = Path(args.input_images)
        input_labels_dir = Path(args.input_labels)
        input_preds_dir = Path(args.input_preds)  

        input_images = sorted(input_images_dir.glob('*.jpg'))
        input_preds = sorted(input_preds_dir.glob('*.txt'))
        input_labels = sorted(input_labels_dir.glob('*.txt'))
        # input_og_preds = sorted(input_og_preds_dir.glob('*.txt'))

        n_regions = 3
        # Array for all regions
        stats, ap, ap_class = [], [], []
        # Region specific arrays
        stats_array = [[] for _ in range(n_regions)]
        # ap_array = [[] for _ in range(n_regions)]
        # ap_class_array = [[] for _ in range(n_regions)]
        
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        names = [ 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]

        # Statistics per image
        for i,v in enumerate(input_labels):
            seen += 1
            #   Create tensor for the label file
            labels_tensor = create_tensor_from_file(v) # = tbox
            if labels_tensor is None: # no lines in the input file
                continue

            #    Create tensor for the predictions
            preds_tensor = create_tensor_from_file(input_preds[i]) # = preds
            predn = preds_tensor.clone()

            # Rectangular, discretized filter
            # filtered_labels_indexes = filter_tensor(labels_tensor, n_regions)
            # filtered_preds_indexes = filter_tensor(preds_tensor, n_regions)

            # Circular filter
            input_img = cv2.imread(str(input_images[i]))
            filtered_labels_indexes = circular_filter(labels_tensor, input_img.shape)
            filtered_preds_indexes = circular_filter(preds_tensor, input_img.shape)

            # Assign all predictions as incorrect
            # correct = torch.zeros(preds_tensor.shape[0], niou, dtype=torch.bool)
            # Region specific arrays
            correct_array = [torch.zeros(preds_tensor[filtered_preds_indexes[r]].shape[0], niou, dtype=torch.bool) for r in range(n_regions)]
            all_correct_array = torch.zeros(preds_tensor.shape[0], niou, dtype=torch.bool)

            # # Testing discretized filter index (heatmaps)
            # highlighted_region = 5
            # print(labels_tensor[filtered_labels_indexes[highlighted_region-1],:])
            # plot_boxes_map(labels_tensor[filtered_labels_indexes[highlighted_region-1],:], n_regions, labels_tensor, highlighted_region)
            # continue

            # ----------------- Compute for all regions together (standard computation) -----------------
            #   Variables/lists/tensors initialisation
            nl = len(labels_tensor) # Number of labels in the tensor
            tcls = labels_tensor[:, 0].tolist() if nl else []  # target class
            detected_count = 0 # Number of labels detected so far

            if len(preds_tensor) == 0: # No predictions
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if nl: # if there are labels
                labels_classes = labels_tensor[:, 0] # = tcls_tensor
                # Per target class
                for cls in torch.unique(labels_classes):
                    ti = (cls == labels_classes).nonzero(as_tuple=False).view(-1) # target indices
                    pi = (cls == preds_tensor[:, 0]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    if pi.shape[0]: # If there is a prediction made
                        # print("Prediction available in this region : ", r, " for class ", cls)
                        # Calculate the IoU between every prediction and target box
                        ious, ind = box_iou(predn[pi,1:5], labels_tensor[ti,1:]).max(1)
                        detected_set = set()

                        # Iterating through prediction indexes with an IoU above 0.5               
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[ind[j]]
                            if d.item() not in detected_set: # if this target has already been detected before, ignore
                                detected_set.add(d.item())
                                detected_count += 1 # Increment detected label count
                                all_correct_array[pi[j]] = ious[j] > iouv 
                                if detected_count == nl:  # all targets already located in image
                                    # print("All targets detected")
                                    break

            # Append to the stats per region after looking at each image
            stats.append((all_correct_array.cpu(), preds_tensor[:, 5].cpu(), preds_tensor[:, 0].cpu(), tcls))
            # mAP per class is then calculated using this stats vector

            # ----------------- Compute for each region individually (radial calculations) -----------------
            for r in range(n_regions):
                # print(" ")
                # print("======== Region ", r+1, " ========")
                # r+1 - represents the region we are looking at

                # Testing - Visualising labels filtered within that region
                # ================================================================
                # in_img = cv2.imread(str(input_images[i]))
                # scale = in_img.shape[1]/in_img.shape[0] # e.g. 16/9, to multiply the x-axis by
                # label_img = apply_tensor_labels(in_img, labels_tensor[filtered_labels_indexes[r]], [0,0,255]) # Red = labels
                # # Draw circular regions to help visualise filtering
                # label_img = cv2.circle(label_img, (int(in_img.shape[1]/2), int(in_img.shape[0]/2)), int(scale*(1/6)*in_img.shape[0]), [255,0,0]) # 1/3
                # label_img = cv2.circle(label_img, (int(in_img.shape[1]/2), int(in_img.shape[0]/2)), int(scale*(0.4)*in_img.shape[0]), [255,0,0]) # 2/3
                # # label_img = cv2.circle(label_img, (int(in_img.shape[1]/2), int(in_img.shape[0]/2)), int(scale*(7/12)*in_img.shape[0]), [255,0,0]) # 1
                # cv2.imshow("Labels in region "+str(r+1), label_img)
                # cv2.waitKey()
                # ================================================================

                #   Variables/lists/tensors initialisation
                # print("Labels tensor : ", labels_tensor)
                # print("Indexed : ", labels_tensor[filtered_labels_indexes[r]], 0)
                nl = len(labels_tensor[filtered_labels_indexes[r]]) # Number of labels in the tensor
                # print("Number of labels : ", nl)
                tcls = labels_tensor[filtered_labels_indexes[r], 0].tolist() if nl else []  # target class
                # print("tcls: ", tcls)
                detected_count = 0 # Number of labels detected so far

                if len(preds_tensor[filtered_preds_indexes[r]]) == 0: # No predictions
                    # print("No predictions found in this region")
                    if nl:
                        stats_array[r].append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                if nl: # if there are labels
                    labels_classes = labels_tensor[filtered_labels_indexes[r], 0] # = tcls_tensor
                    # print("Labels classes : ", labels_classes)
                    for cls in torch.unique(labels_classes):
                        # print("==== Looking at class : ", cls, " ====")
                        ti = (cls == labels_classes).nonzero(as_tuple=False).view(-1) # target indices
                        # print("ti: ", ti)
                        pi = (cls == preds_tensor[filtered_preds_indexes[r], 0]).nonzero(as_tuple=False).view(-1)  # prediction indices
                        # print("pi : ", pi)

                        if pi.shape[0]: # If there is a prediction made
                            # Calculate the IoU between every prediction and target box
                            ious, ind = box_iou(preds_tensor[filtered_preds_indexes[r]][pi,1:5], labels_tensor[filtered_labels_indexes[r]][ti,1:]).max(1)
                            detected_set = set()

                            # Iterating through prediction indexes with an IoU above 0.5               
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[ind[j]]
                                if d.item() not in detected_set: # if this target has already been detected before, ignore
                                    detected_set.add(d.item())
                                    detected_count += 1 # Increment detected label count
                                    # pi[j] = index of a prediction bounding box
                                    # print("ious[j] > iouv: ", ious[j] > iouv) # This checks if that IoU is above all IoU minimums and returns
                                    # an array of True or False per value
                                    correct_array[r][pi[j]] = ious[j] > iouv  # iou_thres is 1xn # THE KEY IS HERE, WHAT IS correct
                                    # The whole row of this is set to True (10 cols for the 10 vals in iouv)
                                    # Setting that index of the predicted bounding box to True if the value is above a threshold
                                    # Correct then comes out showing which predictions are accurate (True along rows) and how accurate
                                    #   (True along columns)
                                    if detected_count == nl:  # all targets already located in image
                                        # print("All targets detected")
                                        break

                # Append to the stats per region after looking at each image
                stats_array[r].append((correct_array[r].cpu(), preds_tensor[filtered_preds_indexes[r], 5].cpu(), preds_tensor[filtered_preds_indexes[r], 0].cpu(), tcls))
                # mAP per class is then calculated using this stats vector

        # Computation for the whole image
        print("================= ALL REGIONS =================")
        p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        # print(stats)
        if len(stats) and stats[0].any():
            # stats - tp (True positives), conf, pred_cls (Predicted classes), target_cls (True classes)
            p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=10)  # number of targets per class
        else:
            print("No predictions found")
            nt = torch.zeros(1)

        # Print results
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        print(s)
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            
        # After having iterated through all images, calculate metrics per region.
        # At this point, all image data has been appended to the stats array.
        for reg in range(n_regions):
            print("================= REGION : ", reg+1, "=================")
            # print("REGION : ", reg+1)
            ap, ap_class = [], []
            p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
            # print(stats_array[reg])
            stats = [np.concatenate(x, 0) for x in zip(*stats_array[reg])]  # to numpy
            # print("stats : ", stats)
            if len(stats) and stats[0].any():
                # stats - tp (True positives), conf, pred_cls (Predicted classes), target_cls (True classes)
                p, r, ap, f1, ap_class = ap_per_class(*stats, names=names)
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=10)  # number of targets per class
            else:
                print("No predictions found")
                nt = torch.zeros(1)

            # Print results
            s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
            print(s)
            pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

            # Print results per class
            if len(stats):
                for i, c in enumerate(ap_class):
                    print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


if __name__ == "__main__":
    main()