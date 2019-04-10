import config as cfg
import torch
import torch.nn as nn


def get_bins_coord_list(x1, y1, x2, y2):
    bins_coord_list = []
    k = cfg.SM_BIN_LENGTH
    width = x2 - x1
    height = y2 - y1
    delta_x = int(width/k)
    delta_y = int(height/k)

    for i in range(k):
        for j in range(k):
            x_start = x1 + i*delta_x
            y_start = y1 + j*delta_y
            x_end = x_start + delta_x
            y_end = y_start + delta_y
            bins_coord_list.append((x_start, y_start, x_end, y_end))

    return bins_coord_list


def map_roi_to_class_probs(roi_x1, roi_y1, roi_x2, roi_y2, score_map):
    k = cfg.SM_BIN_LENGTH
    bins_coord_list = get_bins_coord_list(roi_x1, roi_y1, roi_x2, roi_y2)
    patch = torch.zeros((cfg.NUM_CLASSES+1, k, k))

    for level, coord in enumerate(bins_coord_list):
        current_bin = score_map[level*(cfg.NUM_CLASSES+1):(level+1)*(cfg.NUM_CLASSES+1), coord[0]:coord[2], coord[1]:coord[3]]
        patch[:, int((level - (level % k))/k), int(level % k)] = current_bin.mean(1).mean(1)

    class_scores = patch.mean(1).mean(1)
    return class_scores.softmax(0)




