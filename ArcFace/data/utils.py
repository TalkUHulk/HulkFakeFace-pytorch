from torchvision import transforms
from PIL import Image, ImageOps
import torch

input_shape = (112, 112)

__all__ = ["trans", "target_trans"]


class CvtColor(object):

    def __init__(self, mode="RGB"):
        assert isinstance(mode, str)
        self.mode = mode

    def __call__(self, img):
        print(type(img))
        if isinstance(img, str):
            img = Image.open(input).convert(self.mode)
        elif isinstance(img, Image.Image):
            img = img.convert(self.mode)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(model={})'.format(self.mode)


trans = transforms.Compose([transforms.CenterCrop(input_shape),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])


def target_trans(target):
    assert isinstance(target, int)
    return torch.tensor(target).long()
