import torch
import math
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageEnhance


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask, flow = sample['image'], sample['label'], sample['flow']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            flow = ImageOps.expand(flow, border=self.padding, fill=0)

        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size  # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            flow = flow.resize((tw, th), Image.BILINEAR)
            return {'image': img,
                    'label': mask,
                    'flow': flow}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = Image.fromarray(mask)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(mask)
            flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'flow': flow,
                'depth': depth}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = sample['label'].astype(np.float32)
        flow = np.array(sample['flow']).astype(np.float32)
        depth = np.array(sample['depth']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        flow /= 255.0
        flow -= self.mean
        flow /= self.std

        depth /= 255.0
        depth -= self.mean
        depth /= self.std

        return {'image': img,
                'label': mask,
                'flow': flow,
                'depth': depth}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        flow = np.array(sample['flow']).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(sample['depth']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(sample['label'].astype(np.float32), -1).transpose((2, 0, 1))
        mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        flow = torch.from_numpy(flow).float()
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask,
                'flow': flow,
                'depth': depth}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        depth = sample['depth']

        img = img.resize(self.size, Image.BILINEAR)
        mask = cv2.resize(mask, self.size, cv2.INTER_NEAREST)
        flow = flow.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow,
                'depth': depth}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        flow = flow.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        #edge = sample['edge']
        assert img.size == mask.size
        assert img.size == flow.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                flow = flow.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
                flow = flow.resize((self.size, self.size), Image.BILINEAR)

                return {'image': img,
                        'label': mask,
                        'flow': flow}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        flow = flow.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}

class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']

        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            flow = flow.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        sample = {'image': img, 'label': mask, 'flow': flow}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        return {'image': img, 'label': mask, 'flow': flow}


'''
class RandomRotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        mode = Image.BICUBIC
        label = Image.fromarray(label)
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            label = label.rotate(random_angle, mode)
            flow = flow.rotate(random_angle, mode)
            depth = depth.rotate(random_angle, mode)
        label = np.array(label)
        return {'image': image,
                'label': label,
                'flow': flow,
                'depth': depth}'''

class RandomRotate(object):
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        mode = Image.BILINEAR  # 对于图像，使用双线性插值
        label_mode = Image.NEAREST  # 对于标签，使用最近邻插值
        # 随机旋转角度
        random_angle = random.uniform(-self.degrees, self.degrees)
        # 旋转图像
        image = image.rotate(random_angle, mode)
        # 旋转标签（使用最近邻插值）
        label = Image.fromarray(label)
        label = label.rotate(random_angle, label_mode)
        label = np.array(label)
        # 旋转光流和深度图
        flow = flow.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        return {'image': image,
                'label': label,
                'flow': flow,
                'depth': depth}

class colorEnhance(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return {'image': image,
                'label': label,
                'flow': flow,
                'depth': depth}


class randomPeper(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        noiseNum = int(0.0015 * label.shape[0] * label.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, label.shape[0] - 1)
            randY = random.randint(0, label.shape[1] - 1)
            if random.randint(0, 1) == 0:
                label[randX, randY] = 0
            else:
                label[randX, randY] = 1
        return {'image': image,
                'label': label,
                'flow': flow,
                'depth': depth}


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        flip_flag = random.randint(0, 1)
        # flip_flag2= random.randint(0,1)
        #left right flip
        if flip_flag == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = Image.fromarray(label)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label)
            flow = flow.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        #top bottom flip
        # if flip_flag2==1:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
        #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': label,
                'flow': flow,
                'depth': depth}

#================================新增======================================
class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.8, 1.2)):
        self.size = size   # (h, w)
        self.scale = scale
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        # 确保都是PIL图像
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        if isinstance(flow, np.ndarray):
            flow = Image.fromarray(flow)
        if isinstance(depth, np.ndarray):
            depth = Image.fromarray(depth)
        # 随机缩放因子
        s = random.uniform(self.scale[0], self.scale[1])
        new_w = int(s * img.width)
        new_h = int(s * img.height)
        # 缩放
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        flow = flow.resize((new_w, new_h), Image.BILINEAR)
        depth = depth.resize((new_w, new_h), Image.BILINEAR)
        # 目标尺寸
        target_h, target_w = self.size   # 注意：self.size是(h,w)
        # 如果缩放后的图像宽高小于目标尺寸，则拉伸
        if new_w < target_w or new_h < target_h:
            # 拉伸到目标尺寸
            img = img.resize((target_w, target_h), Image.BILINEAR)
            mask = mask.resize((target_w, target_h), Image.NEAREST)
            flow = flow.resize((target_w, target_h), Image.BILINEAR)
            depth = depth.resize((target_w, target_h), Image.BILINEAR)
            # 此时，图像大小已经是(target_w, target_h)，所以直接使用
            # 将mask转换为numpy数组，其他保持PIL图像
            return {
                'image': img,
                'label': np.array(mask),
                'flow': flow,
                'depth': depth
            }
        else:
            # 随机裁剪
            x = random.randint(0, new_w - target_w)
            y = random.randint(0, new_h - target_h)
            box = (x, y, x+target_w, y+target_h)
            img = img.crop(box)
            mask = mask.crop(box)
            flow = flow.crop(box)
            depth = depth.crop(box)
            return {
                'image': img,
                'label': np.array(mask),   # 转换为numpy数组
                'flow': flow,
                'depth': depth
            }

class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        depth = sample['depth']

        # 亮度调整
        if self.brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # 对比度调整
        if self.contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # 饱和度调整
        if self.saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - self.saturation),
                1 + self.saturation
            )
            img = ImageEnhance.Color(img).enhance(saturation_factor)

        # 色相调整（需要转换为HSV空间）
        if self.hue > 0:
            hue_factor = random.uniform(-self.hue, self.hue)
            img = img.convert('HSV')
            channels = list(img.split())
            hue = channels[0].point(lambda i: (i + hue_factor * 255) % 255)
            img = Image.merge('HSV', (hue, channels[1], channels[2])).convert('RGB')

        return {
            'image': img,
            'label': mask,
            'flow': flow,
            'depth': depth
        }

class RandomGaussianBlur(object):
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img = np.array(sample['image'])

        # 确保核大小为奇数
        kernel_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1

        # 应用高斯模糊
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # 转换回PIL图像
        img = Image.fromarray(img)

        return {'image': img,
                'label': sample['label'],
                'flow': sample['flow'],
                'depth': sample['depth']
        }

class RandomGaussianNoise(object):
    def __init__(self, mean=0, sigma=0.03):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        flow = sample['flow']
        depth = sample['depth']
        # 如果随机数大于0.5，则添加噪声
        if random.random() < 0.5:
            # 将图像转换为numpy数组
            img_np = np.array(image).astype(np.float32)
            # 生成高斯噪声
            noise = np.random.normal(self.mean, self.sigma, img_np.shape).astype(np.float32)
            img_noised = img_np + noise * 255
            # 将值限制在0-255
            img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_noised)
        # 对其他模态（如flow、depth）也可以同样添加噪声，但这里我们只对RGB图像添加
        return {'image': image,
                'label': label,
                'flow': flow,
                'depth': depth}
    
class RandomApply(object):
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p
    def __call__(self, sample):
        if random.random() < self.p:
            for t in self.transforms:
                sample = t(sample)
        return sample



