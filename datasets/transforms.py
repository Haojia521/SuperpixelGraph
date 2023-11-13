import random
import cv2
import numpy as np


class Identity:
    def __call__(self, x, mask=None):
        return x, mask


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class VerticalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class RandomFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 2)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class Transpose:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                mask = mask.transpose(1, 0)
        return img, mask


class Rotate:
    def __init__(self, limit=90, prob=.5):
        self.prob = prob
        self.limit = limit

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            angle = random.uniform(-self.limit, self.limit)

            height, width = img.shape[0:2]
            mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            img = cv2.warpAffine(img, mat, (height, width),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask, mat, (height, width),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class CenterCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        assert h >= self.height and w >= self.width

        dx = (h-self.height)//2
        dy = (w-self.width)//2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = img[y1:y2, x1:x2, :]
        if mask is not None:
            mask = mask[y1:y2, x1:x2]

        return img, mask


class RandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        h, w, c = img.shape
        assert h >= self.height and w >= self.width

        y1 = random.randint(0, h - self.height)
        y2 = y1 + self.height
        x1 = random.randint(0, w - self.width)
        x2 = x1 + self.width

        img = img[y1:y2, x1:x2, :]
        if mask is not None:
            mask = mask[y1:y2, x1:x2]

        return img, mask


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img, mask=None):
        torch_img = np.moveaxis(img / 255., -1, 0).astype(np.float32)
        if mask is not None:
            mask = np.expand_dims(mask, axis=0).astype(np.float32)
            return torch_img, mask
        return torch_img
