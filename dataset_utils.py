from torchvision.datasets import VOCDetection
import config
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


def denormalize(img_tensor, mean_list, std_list):
    mean = torch.tensor(mean_list, dtype=torch.float32)
    std = torch.tensor(std_list, dtype=torch.float32)
    return img_tensor.div_(std[:, None, None]).sub_(mean[:, None, None])


def get_data_loader():
    # Condition for pretrained resnet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = VOCDetection(r'E:/ML/Datasets/Pascal/',
                           year='2007',
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate)


def show_bbox_batch(batch):
    # Inverse mean and std of normalization operation
    mean = [-0.485, -0.456, -0.406]
    std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
    images_batch, annotation_batch = batch[0], batch[1]
    iterations = len(images_batch)

    fig = plt.figure()

    for i in range(iterations):
        img, annot = np.array(transforms.ToPILImage()(denormalize(images_batch[i], mean, std))), annotation_batch[i]
        ax = plt.subplot(1, iterations, i+1)
        plt.tight_layout()
        plt.imshow(img)
        boxes = annot['annotation']['object']

        if type(boxes) != list:
            boxes = [boxes]

        for box in boxes:
            bbox = box['bndbox']
            x = int(bbox['xmin'])
            y = int(bbox['ymin'])
            width = int(bbox['xmax']) - x
            height = int(bbox['ymax']) - y
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.pause(0.001)

        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    loader = get_data_loader()
    for sample in loader:
        show_bbox_batch(sample)
        break
