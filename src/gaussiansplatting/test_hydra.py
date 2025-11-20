import hydra


@hydra.main(version_base="1.2", config_path="gs_config", config_name="train.yaml")
def test_hydra_config(cfg) -> None:
    # print the path where the .py is located
    print(f"Current script path: {hydra.runtime.cwd ()}")


if __name__ == "__main__":
    test_hydra_config()
