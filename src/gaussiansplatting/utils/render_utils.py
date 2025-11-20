import iio
import torchvision


def save_img_between_0and1(name, img, save_png=True):
    """Save an image tensor to a file, knowing the tensor is in 0 and 1 .

    Args:
        name (str): The name of the output file.
        img (torch.Tensor): The image tensor to save. Should be in the range [0, 1].
    """
    if name.endswith(".png"):
        name = name.split(".png")[0]
    if name.endswith(".iio"):
        name = name.split(".iio")[0]

    iio.write(name + ".iio", img.permute(1, 2, 0).cpu().numpy())
    if save_png:
        torchvision.utils.save_image(img, name + ".png")
    return img


def save_img_notscaled(name, img, save_png=True):
    """Save an image tensor to a file, knowing the tensor is not scaled.

    Args:
        name (str): The name of the output file.
        img (torch.Tensor): The image tensor to save. Should be in the range [0, 255].
    """
    if name.endswith(".png"):
        name = name.split(".png")[0]
    if name.endswith(".iio"):
        name = name.split(".iio")[0]

    iio.write(name + ".iio", img.cpu().numpy())
    if save_png:
        torchvision.utils.save_image(img, name + ".png")
    return img
