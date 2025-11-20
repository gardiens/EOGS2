from scene.dataset_readers.dataset_affine import *
from scene.dataset_readers.dataset_utils import MSSceneInfo


def split_MS_traintest_caminfo(path, cam_infos, eval):
    if eval:
        with open(os.path.join(path, "train.txt"), "r") as trainsplit:
            trainsplit = trainsplit.read().splitlines()
            trainsplit = [x.replace(".json", "") for x in trainsplit]
        with open(os.path.join(path, "test.txt"), "r") as testsplit:
            testsplit = testsplit.read().splitlines()
            testsplit = [x.replace(".json", "") for x in testsplit]
        train_cam_infos = []
        test_cam_infos = []
        for caminfo in cam_infos[:-1]:
            if caminfo.image_name in trainsplit:
                train_cam_infos.append(caminfo)
            elif caminfo.image_name in testsplit:
                test_cam_infos.append(caminfo)
            else:
                print(
                    f"the image {caminfo.image_name} is not in train or test split! We skip it for now"
                )
                # raise RuntimeError("Image not in train or test split!")

        # add the last camera (perfectly nadir) to the test
        test_cam_infos.append(cam_infos[-1])

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    return train_cam_infos, test_cam_infos


def readMSAffineSceneInfo(
    path,
    images_path: "dict",
    eval,
    camera_params: "CameraParams",
    load_image_type: "dict",
    need_rescale: bool,
    target_density: float,
    scale_factor_z: float = 1.0,
    modelparams: "ModelParams" = None,
) -> "SceneInfo":
    with open(os.path.join(path, "affine_models.json"), "r") as metadatas:
        metadatas: "dict" = json.load(metadatas)

    keys = metadatas.keys()
    # cam_infos={key:[] for key in keys}
    cam_infos = []
    assert (
        "msi" in keys
    ), "MSI camera info not found in the metadata, are you sure you wanna use MSI camera info?"
    # check that images path and metadata keys match
    assert (
        set(images_path.keys()) == set(metadatas.keys())
    ), "Images path keys do not match metadata keys, got images_path keys: {}, metadata keys: {}".format(
        images_path.keys(), metadatas.keys()
    )
    assert (
        load_image_type["msi"] == True
    ), f"You need to load MSI images, but you set load_image_type['msi']={load_image_type['msi']}"

    nb_images = len(metadatas["msi"])
    for idx in range(nb_images):
        caminfokey = {key: [] for key in keys}

        for key in keys:
            if not load_image_type[
                key
            ]:  # we skip the camera that we don't want to load
                continue
            metadata = metadatas[key][idx]
            if "sun_model" in metadata:
                load_sun = True
            else:
                load_sun = False  #!
            caminfo, min_world, max_world = load_caminfo(
                metadata=metadata,
                load_sun=load_sun,
                images=images_path[key],
                camera_params=camera_params,
                need_rescale=need_rescale,
            )
            caminfokey[key] = caminfo
        # this is the MSAffineCameraInfo that will encompass all the affine cameras
        caminfokey = MSAffineCameraInfo(
            images=caminfokey,
            is_reference_camera=False,
            image_name=caminfokey["msi"].image_name,
            uid=caminfokey["msi"].uid,
        )
        cam_infos.append(caminfokey)
    # split in train test caminfo
    train_cam_infos, test_cam_infos = split_MS_traintest_caminfo(
        path=path, cam_infos=cam_infos, eval=eval
    )
    # Setting the first train camera as the reference camera
    train_cam_infos[0].is_reference_camera = True

    # initialize the point clouds
    ply_path = os.path.join(path, "points3d.ply")

    if isinstance(modelparams.initialize_pcd.input_ply_name, type(None)):
        print("we initialize the pcd as a cloud")
        max_world[2] = max_world[2] * scale_factor_z
        pcd, xyz = initalize_pcd(
            root_path=path,
            metadata=metadata,
            max_world=max_world,
            min_world=min_world,
            ply_path=ply_path,
            target_density=target_density,
        )
    else:
        print("we initialize the pcd from the given ply name")
        print("the input ply path is", ply_path)
        input_name = modelparams.initialize_pcd.input_ply_name
        ply_path = os.path.join(path, f"{input_name}.ply")
        pcd, xyz = initalize_pcd_fromtoaffine(ply_path=ply_path)

    radius = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)
    radius = np.max(radius) * 2

    # The radius variable will be used for densification strategies but also for scaling the spatial_lr
    # 100 è troppo
    # 10 è ok
    # 1 è ok
    # radius = radius * 10

    scene_info = MSSceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization={
            "radius": radius,
            "scale": metadata["model"]["scale"],
            "center": metadata["model"]["center"],
            "n": metadata["model"]["n"],
            "l": metadata["model"]["l"],
        },
        ply_path=ply_path,
    )
    return scene_info
