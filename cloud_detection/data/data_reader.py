import numpy as np
import skimage

from cloud_detection.augmentation.augmentations import flipping_image_and_mask, rotate_clk_image_and_mask, \
    rotate_cclk_image_and_mask, zoom_image_and_mask


class DataReader(object):

    def __init__(self, images, annotations, image_size=384, augment=False, test=False, max_possible_input_value=65535):
        self.annotations = annotations
        self.images = images
        self.image_size = image_size
        self.augment = augment
        self.index = range(len(self.annotations))
        self.test = test
        self.max_possible_input_value = max_possible_input_value

    def __len__(self):
        return len(self.annotations)

    def iteration(self):
        for i in self.index:
            yield self[i]

    def __getitem__(self, index):
        file = self.images[index]

        image_red = skimage.io.imread(file[0])
        image_green = skimage.io.imread(file[1])
        image_blue = skimage.io.imread(file[2])
        image_nir = skimage.io.imread(file[3])

        image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)
        image = skimage.transform.resize(
            image,
            (self.image_size, self.image_size),
            preserve_range=True,
            mode='symmetric'
        )
        if self.test:
            return image
        else:
            mask = self.annotations[index]
            mask = skimage.io.imread(mask)
            mask = skimage.transform.resize(
                mask,
                (self.image_size, self.image_size),
                preserve_range=True,
                mode='symmetric'
            )
            if self.augment:
                random_flip = np.random.randint(2, dtype=int)
                random_rotate_clk = np.random.randint(2, dtype=int)
                random_rotate_cclk = np.random.randint(2, dtype=int)
                random_zoom = np.random.randint(2, dtype=int)

                if random_flip == 1:
                    image, mask = flipping_image_and_mask(image, mask)

                if random_rotate_clk == 1:
                    image, mask = rotate_clk_image_and_mask(image, mask)

                if random_rotate_cclk == 1:
                    image, mask = rotate_cclk_image_and_mask(image, mask)

                if random_zoom == 1:
                    image, mask = zoom_image_and_mask(image, mask)

            mask = mask[..., np.newaxis]
            mask /= 255
            image /= self.max_possible_input_value
            return image, mask
