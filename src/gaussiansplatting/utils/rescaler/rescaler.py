import torch

try:
    import kornia

    use_kornia = True
except:
    use_kornia = False
    pass


def compute_min_vals(image):
    c, h, w = image.shape
    min_vals = image.view(c, -1).min(dim=1)[0]
    return min_vals


def compute_max_vals(image):
    c, h, w = image.shape
    max_vals = image.view(c, -1).max(dim=1)[0]
    return max_vals


class base_rescaler:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def setup(self, list_cam):
        return self


class clamper(base_rescaler):
    def __init__(self, min_val=0, max_val=1):
        # super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class rescale_wrt_firstimage(base_rescaler):
    def __init__(self):
        pass

    def setup(self, list_cam):
        main_cam = None
        for cam in list_cam:
            if cam.is_reference_camera:
                main_cam = cam
                break
        assert main_cam is not None, f"we didn't find the reference camera"
        ref_gt_image = main_cam.original_image
        # Compute per-channel min and max
        # Flatten spatial dimensions before reduction
        self.min_vals = compute_min_vals(ref_gt_image)
        self.max_vals = compute_max_vals(ref_gt_image)

        print("self.min_vals:", self.min_vals.flatten())
        print("self.max_vals:", self.max_vals.flatten())
        print("self.min_als=", self.min_vals)
        return self

    def forward(self, x):
        assert (
            self.min_vals is not None and self.max_vals is not None
        ), "Call setup() before forward()"
        return (x - self.min_vals[:, None, None]) / (
            self.max_vals[:, None, None] - self.min_vals[:, None, None] + 1e-8
        )


class standard_rescaler(base_rescaler):
    def __init__(self):
        pass

    def forward(self, x):
        min_vals = compute_min_vals(x)
        max_vals = compute_max_vals(x)

        return (x - min_vals[:, None, None]) / (
            max_vals[:, None, None] - min_vals[:, None, None] + 1e-8
        )


class histogram_equalizer(base_rescaler):
    def __init__(self):
        from torchvision.transforms.functional import equalize

        self.heq = equalize

    def forward(self, x):
        # convert to uint8 image
        x = x * 255.0
        x = x.to(torch.uint8)
        return self.heq(x) / 255.0


class identity_rescaler(base_rescaler):
    def __init__(self):
        pass

    def forward(self, x):
        return x




class CLAHE_rescaler(base_rescaler):
    def __init__(self):
        self.input_rescaler = standard_rescaler()
        self.grid_size = (8, 8)
        self.clip_limit = 2.0
        if not use_kornia:
            raise ImportError("kornia is not installed, cannot use CLAHE_rescaler")

    def forward(self, x):
        # kornia expect the image to be in format 0,1 or it crashes

        x = self.input_rescaler.forward(x)
        return kornia.enhance.equalize_clahe(
            x, clip_limit=self.clip_limit, grid_size=self.grid_size
        )


def load_rescaler(rescaler_name):
    if rescaler_name == "standard_rescaler":
        return standard_rescaler()
    elif rescaler_name == "rescale_wrt_firstimage":
        return rescale_wrt_firstimage()
    elif rescaler_name == "clamper":
        return clamper()
    elif rescaler_name == "histogram_equalizer":
        return histogram_equalizer()
    elif rescaler_name == "CLAHE_rescaler":
        return CLAHE_rescaler()
    elif rescaler_name == "identity":
        return identity_rescaler()
    elif rescaler_name == "clamper":
        return clamper()
    else:
        raise ValueError(f"Unknown rescaler name: {rescaler_name}")


def perform_rescaling_listcam(list_cam, rescaler):
    rescaler = rescaler.setup(list_cam)
    for cam in list_cam:
        original_image = cam.original_image
        # print("original image hsape",original_image.shape)
        # print("original max and min",original_image.max(),original_image.min())
        # print("output of rescaler is",rescaler.forward(original_image).shape)
        cam.original_image = rescaler.forward(original_image)
        # print("after rescaling the max and min",cam.original_image.max(),cam.original_image.min())
    return list_cam


def perform_rescaling(train_cameras, rescaler, args: "ModelParams"):
    list_cams = []

    if args.load_pan:
        list_pan_cam = [cam.get_pan_cameras() for cam in train_cameras]
        list_cams.append(list_pan_cam)

    if args.load_msi:
        list_msi_cam = [cam.get_msi_cameras() for cam in train_cameras]
        list_cams.append(list_msi_cam)
    for list_cam in list_cams:
        perform_rescaling_listcam(list_cam, rescaler)
    return train_cameras
