import matplotlib.pyplot as plt


def plot_torch_img(img, title, savefig: bool = False, path: str = None):
    # save a torch img
    if img.dim() == 3:
        plt.imshow(img.detach().cpu().numpy().transpose(1, 2, 0))
    elif img.dim() == 2:
        plt.imshow(img.detach().cpu().numpy(), cmap="gray")
    plt.title(title)
    if savefig:
        if path is None:
            raise ValueError("Path must be provided if savefig is True")
        plt.savefig(path)
    plt.show()
    return
