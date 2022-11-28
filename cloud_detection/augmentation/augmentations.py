import numpy as np
import skimage


def rotate_clk_image_and_mask(image, mask):
    angle = np.random.choice((4, 6, 8, 10, 12, 14, 16, 18, 20))
    image_o = skimage.transform.rotate(image, angle, resize=False, preserve_range=True, mode='symmetric')
    mask_o = skimage.transform.rotate(mask, angle, resize=False, preserve_range=True, mode='symmetric')
    return image_o, mask_o


def rotate_cclk_image_and_mask(image, mask):
    angle = np.random.choice((-20, -18, -16, -14, -12, -10, -8, -6, -4))
    image_o = skimage.transform.rotate(image, angle, resize=False, preserve_range=True, mode='symmetric')
    mask_o = skimage.transform.rotate(mask, angle, resize=False, preserve_range=True, mode='symmetric')
    return image_o, mask_o


def flipping_image_and_mask(image, mask):
    image_o = np.flip(image, axis=1)
    mask_o = np.flip(mask, axis=1)
    return image_o, mask_o


def zoom_image_and_mask(image, mask):
    zoom_factor = np.random.choice((1.2, 1.5, 1.8, 2, 2.2, 2.5))  # Currently doesn't have zoom out!
    height, width = image.shape[:2]

    # Width and height of the zoomed image.
    zoomed_height = int(np.round(zoom_factor * height))
    zoomed_width = int(np.round(zoom_factor * width))

    image = skimage.transform.resize(image, (zoomed_height, zoomed_width), preserve_range=True, mode='symmetric')
    mask = skimage.transform.resize(mask, (zoomed_height, zoomed_width), preserve_range=True, mode='symmetric')
    region = np.random.choice((0, 1, 2, 3, 4))

    # Zooming out.
    if zoom_factor <= 1:
        out_image = image
        out_mask = mask

    # Zooming in.
    else:
        # Bounding box of the clipped region within the input array.
        if region == 0:
            out_image = image[0:height, 0:width]
            out_mask = mask[0:height, 0:width]
        if region == 1:
            out_image = image[0:height, zoomed_width - width:zoomed_width]
            out_mask = mask[0:height, zoomed_width - width:zoomed_width]
        if region == 2:
            out_image = image[zoomed_height - height:zoomed_height, 0:width]
            out_mask = mask[zoomed_height - height:zoomed_height, 0:width]
        if region == 3:
            out_image = image[zoomed_height - height:zoomed_height, zoomed_width - width:zoomed_width]
            out_mask = mask[zoomed_height - height:zoomed_height, zoomed_width - width:zoomed_width]
        if region == 4:
            margin_height = height // 2
            margin_width = width // 2
            out_image = image[(zoomed_height // 2 - margin_height):(zoomed_height // 2 + margin_height),
                        (zoomed_width // 2 - margin_width):(zoomed_width // 2 + margin_width)]
            out_mask = mask[(zoomed_height // 2 - margin_height):(zoomed_height // 2 + margin_height),
                       (zoomed_width // 2 - margin_width):(zoomed_width // 2 + margin_width)]

    # To make sure the output is in the same size of the input.
    image_o = skimage.transform.resize(out_image, (height, width), preserve_range=True, mode='symmetric')
    mask_o = skimage.transform.resize(out_mask, (height, width), preserve_range=True, mode='symmetric')
    return image_o, mask_o
