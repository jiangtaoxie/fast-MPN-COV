from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def preprocess_strategy(dataset):
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
    elif dataset.startswith('Cars'):
        train_transforms = transforms.Compose([
            transforms.Resize((448,448)),
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
    elif dataset.startswith('ImageNet'):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise KeyError("=> transform method of '{}' does not exist!".format(dataset))
    return train_transforms, val_transforms
