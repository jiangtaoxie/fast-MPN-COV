from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import torch
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def center_crop_with_flip(img, size, vertical_flip=False):
    crop_h, crop_w = size
    first_crop = F.center_crop(img, (crop_h, crop_w))
    if vertical_flip:
        img = F.vflip(img)
    else:
         img = F.hflip(img)
    second_crop = F.center_crop(img, (crop_h, crop_w))
    return (first_crop, second_crop)

class CenterCropWithFlip(object):
    """Center crops with its mirror version.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return center_crop_with_flip(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, vertical_flip={1})'.format(self.size, self.vertical_flip)

def preprocess_strategy(dataset):
    evaluate_transforms = None
    if dataset.startswith('CUB'):
        train_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize(448),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('Aircraft'):
        train_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('Cars'):
        train_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize((448,448)),
            CenterCropWithFlip(448),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif dataset.startswith('ImageNet'):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        evaluate_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    else:
        raise KeyError("=> transform method of '{}' does not exist!".format(dataset))
    return train_transforms, val_transforms, evaluate_transforms
