from .to_affine import *
import hydra
from omegaconf import OmegaConf as Omegaconf
import torch


def create_virtual_cam(
    output_metadatas, azimut_angles, altitude_angle, img_names: "list", idx=0
):
    # Now we samples affine camera in a circle around the nadir camera
    print("idx is ", idx)
    for key in output_metadatas.keys():
        print(
            "the metadata image name of the ref_cam  is ",
            output_metadatas[key][idx]["img"],
        )
        break

    for key in output_metadatas.keys():
        for k in range(len(azimut_angles)):
            metadata = deepcopy(output_metadatas[key][idx])

            azimut = azimut_angles[k]
            # Compute the vector direction corresponding to the azimut and altitude angles
            x = torch.cos(azimut) * torch.cos(altitude_angle)
            y = torch.sin(azimut) * torch.cos(altitude_angle)
            z = torch.sin(altitude_angle)
            d = torch.stack([x, y, z], dim=0)
            d = d / d.norm()
            # construct the right A
            affine_camera = torch.eye(4, 4)
            affine_camera[:3, :3] = torch.tensor(metadata["model"]["coef_"])
            affine_camera[:3, -1] = torch.tensor(metadata["model"]["intercept_"])
            affine_camera = affine_camera.float().T
            A = affine_camera[:3, :3].T
            b = affine_camera[3, :3]

            q = A @ d
            q = q / q[-1]
            myM = torch.eye(3)
            myM[:2, 2] = -q[:2]

            new_A = myM @ A
            new_b = (torch.eye(3, device=myM.device) - myM) @ A @ np.array(
                metadata["centerofscene_UTM"]
            ) + b

            view2cc = torch.linalg.inv(myM.double()).float()
            new_A = new_A.numpy()
            new_b = new_b.numpy()

            new_affine = np.eye(4, 4)
            new_affine[:3, :3] = new_A
            new_affine[:3, -1] = new_b
            new_metadata = deepcopy(metadata)
            new_metadata["img"] = img_names[k]
            new_metadata["model"]["coef_"] = new_affine[:3, :3].tolist()
            new_metadata["model"]["intercept_"] = new_affine[:3, -1].tolist()
            new_metadata["virtual_camera"] = True
            # we have to modify the cam2sun too. Come from scaling mat
            #
            scalingmat = torch.eye(4, device="cpu")
            scalingmat[0, 0] = 1 / 2
            scalingmat[1, 1] = 1 / 2
            camera_to_sun = torch.tensor(metadata["sun_model"]["camera_to_sun"])
            cc2sun = scalingmat[:3, :3] @ camera_to_sun
            # camera_to_sun = torch.tensor(metadata["sun_model"]["camera_to_sun"])
            # cc2sun= camera_to_sun
            # * Copy pasted from get_sun_camera+ scaling_mat

            cam2virt = cc2sun @ view2cc
            new_metadata["sun_model"]["camera_to_sun"] = cam2virt.cpu().numpy().tolist()
            # we have to modify the world_view transform and full proj transform as well.

            output_metadatas[key].append(new_metadata)

    return output_metadatas


@hydra.main(
    version_base="1.2",
    config_path="config",
    config_name="main_affine_video.yaml",
)
def hydra_main2(cfg):
    whole_path = os.path.join(cfg.output_expe_path)
    print("we are loading the cfg from ", whole_path)
    angle_number = cfg.angle_number  # intially 200
    azimut_angles = torch.linspace(0, 2 * torch.pi, angle_number)
    # azimut_angles = torch.linspace(0, 2 * torch.pi, 3)
    altitude_angle = torch.deg2rad(torch.tensor(45.0))
    img_names = [f"virtual_cam_{i:03d}.png" for i in range(len(azimut_angles))]
    # load the config that is inside whole_path/cfg.yaml
    cfg_expe = Omegaconf.load(f"{whole_path}/cfg_args.yaml")
    source_path = cfg_expe.source_path

    # Load the actual cfg_to_affine
    print("the original source path for to_affine is ", source_path)
    cfg_to_affine = Omegaconf.load(f"{source_path}/cfg.yaml")

    # Run the same to_affine process but with save_data=False
    cfg_to_affine.save_data = False
    SCENE_METADATA = os.path.join(
        os.path.expanduser(
            os.path.join(
                cfg_to_affine.root_dir,
                cfg_to_affine.scene_name,
            )
        )
    )
    output_metadata = hydra_main(cfg_to_affine)

    # Now we add to the output_metadatas the virtual_cameras

    output_metadata = create_virtual_cam(
        output_metadata,
        idx=cfg.idx,
        azimut_angles=azimut_angles,
        altitude_angle=altitude_angle,
        img_names=img_names,
    )

    # we save the new output metadata exactly as to_affine would do but add the prefix _video

    dataset_destination_path = cfg_to_affine.dataset_destination_path
    scene_name = cfg_to_affine.scene_name
    prefix = cfg_to_affine.prefix
    DATASET_DESTINATION = Path(
        os.path.expanduser(
            os.path.join(
                dataset_destination_path, f"{scene_name}_{prefix}_{cfg.video_prefix}"
            )
        )
    )
    TRAINTEST_DIR = os.path.expanduser(os.path.join(cfg.traintest_dir, scene_name))
    print("we have a number of different images ", len(img_names))
    print("save the metadata in", DATASET_DESTINATION)
    os.makedirs(DATASET_DESTINATION, exist_ok=True)

    # change the test.txt to contain only the new images
    with open(f"{DATASET_DESTINATION}/test.txt", "w") as f:
        for name in img_names:
            f.write(f"{name}\n")
        # save the json
        with open(f"{DATASET_DESTINATION}/affine_models.json", "w") as f:
            json.dump(output_metadata, f, indent=4)
        # Copy also the test.txt and train.txt files
        os.system(f"cp {TRAINTEST_DIR}/train.txt {DATASET_DESTINATION}")
        print("we saved in ", DATASET_DESTINATION)
        if not isinstance(cfg, type(None)):
            with open(os.path.join(DATASET_DESTINATION, "cfg.yaml"), "w") as f:
                OmegaConf.save(cfg, f)


if __name__ == "__main__":
    # tyro is only used for CLI
    hydra_main2()
