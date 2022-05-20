from torch import nn
from torchvision import transforms

__all__ = ['get_trans']


def get_trans(size, split='train'):
    assert split in ['train', 'val', 'test']
    return globals()['_' + split](size)


def _train(size):
    kernel_size = int((size // 20) * 2) + 1
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size, [0.1, 2])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _val(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _test(size):
    return _val(size)