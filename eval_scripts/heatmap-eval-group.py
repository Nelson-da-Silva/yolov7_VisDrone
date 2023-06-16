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
# parser.add_argument('--input_dir', '-d')
parser.add_argument('--input_images', '-i')
parser.add_argument('--input_labels', '-l')
parser.add_argument('--input_og_preds', '-op')
parser.add_argument('--input_preds', '-p')
parser.add_argument('--results_dir', '-r')

args = parser.parse_args()

def apply_labels(input_image, label_file, colour):
    label_data = open(label_file)
    h, w, _ = input_image.shape
    for i,line in enumerate(label_data): # Iterate through each line of the annotation file
        # print(line)
        if i==128:
            return input_image
        if len(line.split(" ")) == 6:
            inst_class, x_centre, y_centre, bbox_w, bbox_h, _ = map(float, line.split(" "))
        else:
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

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def round_up(x, a):
    # print("x/a ", (x/a))
    return math.ceil(x/a)*a

def discretize_dimensions(input_tensor_line, x_dim, y_dim):
    # Get dimensions of the label
    x_left = round(round_up(float(input_tensor_line[1]),1/x_dim)*x_dim) - 1
    # print(labels_tensor[d][0][1], " mapped to ", x_left)
    x_right = round(round_up(float(input_tensor_line[3]),1/x_dim)*x_dim) - 1
    # print(labels_tensor[d][0][3], " mapped to ", x_right)
    # Appending to the heatmap should be reversed vertically
    #   since (0,0) is at the top left
    y_left = round(round_up(input_tensor_line[2],1/y_dim)*y_dim) - 1
    # print(labels_tensor[d][0][2], " mapped to ", y_left)
    y_right = round(round_up(input_tensor_line[4],1/y_dim)*y_dim) - 1
    return x_left, x_right, y_left, y_right

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

def main():
    if args.input_labels and args.input_og_preds and args.input_preds:
        # Expected input directory is the parent directory for the images
        #   in the subset of that dataset
        # input_images_dir = Path(args.input_dir+"/images/")
        # input_images_dir = Path(args.input_images)
        input_labels_dir = Path(args.input_labels)
        # Expect predictions to include confidence
        input_preds_dir = Path(args.input_preds)
        input_og_preds_dir = Path(args.input_og_preds)
        if args.results_dir:
            results_dir = Path(args.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

        # input_images    = sorted(input_images_dir.glob('*.jpg'))
        input_labels    = sorted(input_labels_dir.glob('*.txt'))
        input_preds     = sorted(input_preds_dir.glob('*.txt'))
        input_og_preds  = sorted(input_og_preds_dir.glob('*.txt'))

        #   Global heatmaps across all images
        hm_xlen, hm_ylen = 18,18 # Heatmap dimensions (originally rectangular at 32:18 (16:9))
        # Average IoU per region
        iou_hm = np.zeros((hm_ylen,hm_xlen)) # IoU heatmap (to accumulate IoU across all images)
        iou_hm_sm = np.zeros((hm_ylen,hm_xlen)) # Sum of boxes with an IoU present (to then divide)
        # Average IoU per region for the original model
        og_iou_hm = np.zeros((hm_ylen,hm_xlen))
        og_iou_hm_sm = np.zeros((hm_ylen,hm_xlen))
        # Regional counters for missed labels
        missed_label_hm = np.zeros((hm_ylen, hm_xlen)) # Map giving the sum of missed labels in an image
        og_missed_label_hm = np.zeros((hm_ylen, hm_xlen))
        label_hm = np.zeros((hm_ylen, hm_xlen)) # Heatmap where labels exist

        # Iterate through each image
        for i,v in enumerate(input_labels):
            # print("LOOKING AT IMAGE NO. ", i+1)
            # Corresponding prediction label file = input_preds[i]
            # Corresponding image file = input_images[i]
            
            #   Visualising boundary boxes
            # in_img = cv2.imread(str(input_images[i]))
            # dist_img = apply_labels(in_img, input_labels[i], [0,0,255]) # Red = labels
            # dist_img = apply_labels(in_img, input_preds[i], [255,0,0]) # Blue = predictions
            # cv2.imshow("Predictions", dist_img)
            # cv2.waitKey()

            #   Create tensor for the label file
            labels_tensor = create_tensor_from_file(v)
            if labels_tensor is None: # no lines in the input file
                continue
            labels_classes = labels_tensor[:, 0] # Classes in the labels

            #    Create tensor for the predictions
            preds_tensor = create_tensor_from_file(input_preds[i])
            #    Create tensor for the predictions for the original model
            og_preds_tensor = create_tensor_from_file(input_og_preds[i])

            #       Variables/lists/tensors initialisation
            nl = len(labels_tensor) # Number of labels in the tensor
            detected_count = 0 # Number of labels detected so far

            # Iterating through classes from the labels
            for cls in torch.unique(labels_classes):
                # print(" ")
                # print("Class : ", cls)
                # Indexes of the labels which belong to this class
                ti = (cls == labels_classes).nonzero(as_tuple=False).view(-1) # target indices
                # print("ti: ", ti)
                # Indexes of the predictions which belong to this class
                pi = (cls == preds_tensor[:, 0]).nonzero(as_tuple=False).view(-1) # prediction indices
                # print("pi: ", pi)
                # Indexes of the original predictions
                og_pi = (cls == og_preds_tensor[:, 0]).nonzero(as_tuple=False).view(-1)

                if pi.shape[0]: # If there is at least one prediction made
                    # Calculate the IoU between every prediction and target box
                    ious, i = box_iou(preds_tensor[pi,1:5], labels_tensor[ti,1:]).max(1)
                    # ious - maximum along each row (get the best IoU pair per prediction)
                    # i - column index per maximum
                    # print("ious: ", ious)
                    # print("i: ", i)
                    detected_set = set()
                    predicted_set = set()

                    # Iterating through prediction indexes with an IoU above 0.5               
                    for j in (ious > 0.5).nonzero(as_tuple=False):
                        # pi[j] returns the prediction bb index from the original preds_tensor
                        # preds_tensor[pi[j]] - associated dimensions info to the prediction bb
                        # ti[i[j]] returns the label bb index from the original labels_classes
                        # labels_tensor[ti[i[j]]] - associated dimension info to the label bb
                        # ious[j] - returns the IoU of the associated box pairing

                        d = ti[i[j]]
                        if d.item() not in detected_set:
                            predicted_set.add(pi[j].item()) # Add prediction to the predicted set
                            detected_set.add(d.item()) # Add label to the detected set
                            detected_count += 1 # Increment detected label count

                            # Convert coordinates to heatmap dimensions
                            x_left, x_right, y_left, y_right = discretize_dimensions(labels_tensor[d][0], hm_xlen, hm_ylen)
                            # print("Plotting from TL : (",x_left,", ",y_left,") and BR : (",x_right,", ",y_right,")")

                            # Fill the associated heatmaps in with the necessary information when a prediction has been matched with a label there
                            for l in range(x_left,x_right+1):
                                for k in range (y_left, y_right+1):
                                    iou_hm[k,l] += ious[j]
                                    iou_hm_sm[k,l] += 1

                            if detected_count == nl:
                                # print("All labels detected")
                                break
                    
                    # ====== Identifying missed targets ======
                    # After looking at the predictions for the class we are currently looking at
                    # print("---")
                    # print("predicted set : ", predicted_set)
                    # print("detected_set : ", list(detected_set))
                    target_list = [int(x) for x in ti]

                    for label in target_list:
                        x_left, x_right, y_left, y_right = discretize_dimensions(labels_tensor[label], hm_xlen, hm_ylen)
                        for l in range(x_left,x_right+1):
                            for k in range (y_left, y_right+1):
                                label_hm[k,l] += 1

                    # print("list of target labels: ", target_list)
                    # print("---")

                    missed_targets = list(set(target_list) - detected_set)
                    # print("missed target labels : ", missed_targets)

                    # Identifying missed labels and summing in the heatmap
                    for label in missed_targets:
                        # print(labels_tensor[label])
                        x_left, x_right, y_left, y_right = discretize_dimensions(labels_tensor[label], hm_xlen, hm_ylen)
                        # print(y_left)
                        # print(y_right)
                        for l in range(x_left,x_right+1):
                            for k in range (y_left, y_right+1):
                                missed_label_hm[k,l] += 1
                    
                # ============ Repeat prediction analysis but for the predictions from the original model ============
                if og_pi.shape[0]: # If there is at least one prediction made
                    detected_count = 0
                    # Calculate the IoU between every prediction and target box
                    ious, i = box_iou(og_preds_tensor[og_pi,1:5], labels_tensor[ti,1:]).max(1)
                    # ious - maximum along each row (get the best IoU pair per prediction)
                    # i - column index per maximum
                    # print("ious: ", ious)
                    # print("i: ", i)
                    detected_set = set()
                    predicted_set = set()

                    # Iterating through prediction indexes with an IoU above 0.5               
                    for j in (ious > 0.5).nonzero(as_tuple=False):
                        # pi[j] returns the prediction bb index from the original preds_tensor
                        # preds_tensor[pi[j]] - associated dimensions info to the prediction bb
                        # ti[i[j]] returns the label bb index from the original labels_classes
                        # labels_tensor[ti[i[j]]] - associated dimension info to the label bb
                        # ious[j] - returns the IoU of the associated box pairing

                        # print("Prediction : ", pi[j])
                        # print(preds_tensor[pi[j]])
                        # print(i[j])
                        d = ti[i[j]]
                        # print("Associated label : ", d)
                        # print(labels_tensor[d])
                        # print("------------")

                        if d.item() not in detected_set:
                            predicted_set.add(og_pi[j].item()) # Add prediction to the predicted set
                            detected_set.add(d.item()) # Add label to the detected set
                            detected_count += 1 # Increment detected label count

                            # Convert coordinates to heatmap dimensions
                            x_left, x_right, y_left, y_right = discretize_dimensions(labels_tensor[d][0], hm_xlen, hm_ylen)
                            # print("Plotting from TL : (",x_left,", ",y_left,") and BR : (",x_right,", ",y_right,")")

                            # Fill the associated heatmaps in with the necessary information
                            for l in range(x_left,x_right+1):
                                for k in range (y_left, y_right+1):
                                    og_iou_hm[k,l] += ious[j]
                                    og_iou_hm_sm[k,l] += 1

                            if detected_count == nl:
                                # print("All labels detected")
                                break
                    
                    # ====== Identifying missed targets ======
                    # After looking at the predictions for the class we are currently looking at
                    # print("---")
                    # print("predicted set : ", predicted_set)
                    # print("detected_set : ", list(detected_set))
                    target_list = [int(x) for x in ti]
                    # print("list of target labels: ", target_list)
                    # print("---")

                    missed_targets = list(set(target_list) - detected_set)
                    # print("missed target labels : ", missed_targets)

                    # Identifying missed labels and summing in the heatmap
                    for label in missed_targets:
                        # print(labels_tensor[label])
                        x_left, x_right, y_left, y_right = discretize_dimensions(labels_tensor[label], hm_xlen, hm_ylen)
                        # print(y_left)
                        # print(y_right)
                        for l in range(x_left,x_right+1):
                            for k in range (y_left, y_right+1):
                                og_missed_label_hm[k,l] += 1

        # ============ Heatmap processing ============
        # Heatmap normalisation and plotting
        for i in range(hm_ylen):
            for j in range(hm_xlen):
                if iou_hm_sm[i,j] != 0:
                    iou_hm[i,j] = iou_hm[i,j]/iou_hm_sm[i,j]
                if og_iou_hm_sm[i,j] != 0:
                    og_iou_hm[i,j] = og_iou_hm[i,j]/og_iou_hm_sm[i,j]
                if label_hm[i,j] != 0:
                    missed_label_hm[i,j] = missed_label_hm[i,j]/label_hm[i,j]
                    og_missed_label_hm[i,j] = og_missed_label_hm[i,j]/label_hm[i,j]

        # Calculate differences in IoU between the model being evaluated and the original model
        diff_iou_hm = iou_hm - og_iou_hm
        # Global if statements across the heatmaps to separate increases and decreases
        inc_hm = np.where(diff_iou_hm >= 0, diff_iou_hm, 0) # Indexes where there was an increase in IoU
        # inc_hm = diff_iou_hm[diff_iou_hm >= 0].reshape(diff_iou_hm.shape)
        # print(inc_hm.shape)
        dec_hm = np.where(diff_iou_hm < 0, -diff_iou_hm, 0) # Indexes where there was a decrease in IoU
        # dec_hm = diff_iou_hm[diff_iou_hm < 0].reshape(diff_iou_hm.shape)

        # Calculate differences in number of missed labels between the models
        diff_labels_missed = og_missed_label_hm - missed_label_hm
        dec_label_hm = np.where(diff_labels_missed >= 0, diff_labels_missed, 0) # Filtering where there was a decrease in missed labels
        inc_label_hm = np.where(diff_labels_missed < 0, -diff_labels_missed, 0) # Filtering where there was an increase in missed labels

        # Plot the heatmaps
        plt.figure(figsize=(12,8))
        plt.subplot(2,2,1)
        plt.title("New model : Average IoU per region") # Constrained to where predictions have been matched
        ax = sns.heatmap(iou_hm, cmap='crest', vmin=0.5, vmax=1)
        sns.heatmap(iou_hm_sm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=iou_hm != 0, cbar=False, ax=ax)
        plt.subplot(2,2,2)
        plt.title("New model : Proportion of undetected labels per region") # Constrained to where labels exist
        ax = sns.heatmap(missed_label_hm, cmap='crest', vmin=0, vmax=1)
        sns.heatmap(iou_hm_sm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=label_hm != 0, cbar=False, ax=ax)
        plt.subplot(2,2,3)
        plt.title("Baseline model : Average IoU per region")
        ax = sns.heatmap(og_iou_hm, cmap='crest', vmin=0.5, vmax=1) # Constrained to where predictions have been matched
        sns.heatmap(og_iou_hm_sm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=og_iou_hm != 0, cbar=False, ax=ax)
        plt.subplot(2,2,4)
        plt.title("Baseline model : Proportion of undetected labels per region") # Constrained to where labels exist
        ax = sns.heatmap(og_missed_label_hm, cmap='crest', vmin=0, vmax=1)
        sns.heatmap(og_iou_hm_sm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=label_hm != 0, cbar=False, ax=ax)

        plt.show()
        # if args.results_dir:
        #     plt.savefig(args.results_dir+"/separate_graphs.jpg")

        # Heatmaps that show comparison to the original model
        plt.figure(figsize=(12,8))
        plt.subplot(2,2,1)
        plt.title("Increases in IoU") # Constrained to where there have been increases in IoU
        ax = sns.heatmap(inc_hm, cmap='crest', vmin=0, vmax=1)
        sns.heatmap(inc_hm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=inc_hm != 0, cbar=False, ax=ax)
        plt.subplot(2,2,2)
        plt.title("Decreases in proportion of missed labels") # Constrained to where there has been a decrease in the number of missed labels
        ax = sns.heatmap(dec_label_hm, cmap='crest', vmin=0, vmax=1)
        sns.heatmap(dec_label_hm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=dec_label_hm != 0, cbar=False, fmt='%d', ax=ax)
        plt.subplot(2,2,3)
        plt.title("Decreases in IoU") # Constrained to where there have been decreases in IoU
        ax = sns.heatmap(dec_hm, cmap='crest', vmin=0, vmax=1)
        sns.heatmap(dec_hm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=dec_hm != 0, cbar=False, ax=ax)
        plt.subplot(2,2,4)
        plt.title("Increases in proportion of missed labels") # Constrained to where there has been an increase in the number of missed labels
        ax = sns.heatmap(inc_label_hm, cmap='crest', vmin=0, vmax=1)
        # ax = sns.heatmap(inc_label_hm, cmap='crest_r', vmin=0, cbar_kws=dict(ticks=range(int(inc_label_hm.min()-1), int(inc_label_hm.max()) + 2)))
        sns.heatmap(inc_label_hm, cmap=plt.get_cmap('binary'), vmin=0, vmax=0, mask=inc_label_hm != 0, cbar=False, ax=ax)

        plt.show()
        # if args.results_dir:
        #     plt.savefig(args.results_dir+"/comparison_graphs.jpg")

    else:
        print("Input directory not provided with the needed flags")

if __name__ == "__main__":
    main()