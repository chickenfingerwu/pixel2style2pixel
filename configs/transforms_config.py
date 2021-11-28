from abc import abstractmethod
import torchvision.transforms as transforms
import numpy as np
from math import floor
from PIL import Image
import torch
import random
from datasets import augmentations


class TransformsConfig(object):

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        pass


class EncodeTransforms(TransformsConfig):

    def __init__(self, opts):
        super(EncodeTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': None,
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class FrontalizationTransforms(TransformsConfig):

    def __init__(self, opts):
        super(FrontalizationTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class SketchToImageTransforms(TransformsConfig):

    def __init__(self, opts):
        super(SketchToImageTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]),
        }
        return transforms_dict


class SegToImageTransforms(TransformsConfig):

    def __init__(self, opts):
        super(SegToImageTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.ToOneHot(self.opts.label_nc),
                transforms.ToTensor()])
        }
        return transforms_dict


class SuperResTransforms(TransformsConfig):

    def __init__(self, opts):
        super(SuperResTransforms, self).__init__(opts)

    def get_transforms(self):
        if self.opts.resize_factors is None:
            self.opts.resize_factors = '1,2,4,8,16,32'
        factors = [int(f) for f in self.opts.resize_factors.split(",")]
        print("Performing down-sampling with factors: {}".format(factors))
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.BilinearResize(factors=factors),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                augmentations.BilinearResize(factors=factors),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict

class FontToKanjiTransform(TransformsConfig):

    def __init__(self, opts):
        super(FontToKanjiTransform, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ElasticDistortion(6, 6, 7)]),
            'transform_source': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]),
        }
        return transforms_dict

class ElasticDistortion:
    def __init__(self, grid_width, grid_height, magnitude):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude
        # self.iter = -1
        # self.freq = distort_freq

    def __call__(self, image):
        """
        Distorts the passed image(s) according to the parameters supplied during
        instantiation, returning the newly distorted image.
        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if random.uniform(0, 1) <= 0.7:
            return image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image).convert('RGB')

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w, h = image.size

        grid_width = self.grid_width
        grid_height = self.grid_height
        magnitude = self.magnitude

        horizontal_tiles = grid_width
        vertical_tiles = grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-magnitude, magnitude)
            dy = random.randint(-magnitude, magnitude)
            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        def do(image):
            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        # augmented_images = []
        #
        # for image in images:
        #     augmented_images.append(do(image))
        image = do(image)
        image.save('process.png')
        # tensor_image = transforms.ToTensor()(image).unsqueeze_(0)
        # tensor_images = torch.cat(tensor_images)
        return image
