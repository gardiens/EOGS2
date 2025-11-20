from clearml.logger import Logger
from omegaconf import DictConfig, OmegaConf


def connect_cfg(
    cfg: DictConfig,
    task: Logger,
    name="main cfg",
    description=" Name of the configuration used for the inference ",
):
    """Log the whole configuration in configuration/configuration Object. Everything is logged but
    it is not easy to read.

    Parameters
    ----------
    cfg : DictConfig
        the config we will load
    task : Logger
        the ClearML task
    name : str, optional
        name that will be displayed in the corresponding windows, by default "main cfg"
    description : str, optional
        the description of the item, by default " Name of the configuration used for the inference "

    Returns
    -------
    task
         the same ClearML task
    """
    container = OmegaConf.to_container(cfg)
    task.connect_configuration(container, name=name, description=description)
    return task


def connect_hyperparams_summary(
    cfg: DictConfig, task: Logger, name="summary hyperparams", resolve=False
):
    """Connect some hyperparameters on CLEARML, we only connect the main important hyperparameters
    ( datamodule and model for now)

    Parameters
    ----------
    cfg : DictConfig
        the dictConfig we are about to log
    task : Logger
        the ClearML Logger
    name : str, optional
        name that will be displayed in configuration/hyperparameters, by default "summary hyperparams"
    resolve : bool, optional
        If yes it removes every evaluation ${} but didn't work for now , by default False
    """
    # KEYS_NOT_LOGGED=["hydra","logger","callbacks","trainer","paths","clearml"]
    KEYS_TO_LOG = ["train", "test", "model", "datamodule", "best_ckpt", "load_model"]
    # OmegaConf.set_struct(cfg_copy, False)
    """for key in KEYS_NOT_LOGGED:
        if key in cfg_copy.keys():
            del cfg_copy[key]"""

    cfg_copy = OmegaConf.masked_copy(cfg, KEYS_TO_LOG)
    if resolve:
        container = OmegaConf.to_container(
            cfg_copy, resolve=True, throw_on_missing=False
        )
    else:
        container = OmegaConf.to_container(cfg_copy)
    """ if resolve:
        container=OmegaConf.to_container(cfg,resolve=True,throw_on_missing=False)"""

    task.connect(container, name=name)
    return


def set_user_properties(cfg: DictConfig, task: Logger):
    """Define the user properties for ClearML that will be displayed on user properties/properties.

    , It logs the name of the model and the corresponding logs path It is hardcoded because we
    don't change it often it tries to log:

    - the name of the model we are evaluating
    - the dataset we will predict on
    - the output_dir of the logs
    Parameters
    ----------
    cfg : DictConfig
        The config we will log
    task : Logger
        The Task from ClearML
    """
    try:
        task.set_user_properties(
            {
                "name": "trained_model",
                "description": "name of the trained model",
                "value": cfg.loading_model.name_run,
            }
        )
    except:
        pass
    try:
        task.set_user_properties(
            {
                "name": "initial_ckpt",
                "description": "name of the initial ckpt",
                "value": cfg.load_model.path_ckpt,
            }
        )
    except:
        pass
    try:
        task.set_user_properties(
            {
                "name": "Target Dataset",
                "description": "name of the dataset we try to predict or train",
                "value": cfg.datamodule._target_,
            }
        )
    except:
        pass
    try:
        task.set_user_properties(
            {
                "name": "log path",
                "description": "path of the log",
                "value": cfg.paths.output_dir,
            }
        )
    except:
        pass


from clearml import Task


def get_safe_name(
    project_name="drone-cloud-point-segmentation", task_name: str = "debug"
):
    """Check if the name is already used in a task or not.

    Output a name like: name if it is the first one of name/integer for the others It is made for
    hydra config first and rely on a configuration of the name task_name/id
    """
    # Check if the name is already used by ClearML or not
    query: list = Task._query_tasks(
        project_name=project_name, task_name=task_name
    )  # Query all value that contains the same project_name and task_name

    number = len(query)  # We assert that this number is not used

    if number == 0:  # Unique identifier
        return task_name
    # Sanity check
    # To be robust against task deletion, we take the last query and we try to see his number ..
    if number > 1:
        try:
            new_number = int(query[-1].name.split("/")[-1]) + 1
        except:
            new_number = 1

    else:
        new_number = 1
    new_name = task_name + "/" + str(max(number, new_number))
    other_task = Task.get_task(project_name=project_name, task_name=new_name)
    if not isinstance(
        other_task, type(None)
    ):  # It should return Nonetype if no task has this name
        raise NotImplementedError(f"the task_name {new_name} is already taken")
    return new_name


def safe_init_clearml(project_name, task_name, *args, **kwargs) -> Task:
    """Properly initiate clearml.

    Args:
        project_name (str): The name of the project.
        task_name (str): The name of the task.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Task: The initialized clearml Task object.
    """
    # * workaround: if you provide some / in task_name then add it in project name
    if "/" in task_name:
        # check the number of / in task_name
        project_name = f"{project_name}/{'/'.join(task_name.split('/')[:-1])}"
        task_name = task_name.split("/")[-1]
    # get a unique name for the task
    new_name = get_safe_name(project_name=project_name, task_name=task_name)

    # init the task
    task = Task.init(project_name=project_name, task_name=new_name, *args, **kwargs)
    return task


def safe_resume_clearml(project_name, task_name, *args, **kwargs) -> Task:
    """Properly initiate clearml.

    Args:
        project_name (str): The name of the project.
        task_name (str): The name of the task.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Task: The initialized clearml Task object.
    """
    # * don't set the random seed
    Task.set_random_seed(None)
    # * workaround: if you provide some / in task_name then add it in project name
    if "/" in task_name:
        # check the number of / in task_name
        project_name = f"{project_name}/{'/'.join(task_name.split('/')[:-1])}"
        task_name = task_name.split("/")[-1]
    # get a unique name for the task
    new_name = get_safe_name(project_name=project_name, task_name=task_name)
    # hack to resume the task
    if len(new_name.split("/")) > 1:
        number = int(new_name.split("/")[-1])
        if number == 1:
            new_name = new_name.split("/")[0]
        else:
            new_name = new_name.split("/")[0] + "/" + str(number - 1)
    else:
        raise NotImplementedError(
            f"We are resuming a task , so the new name should have a /, got {new_name}"
        )
    # init the task

    task = Task.init(
        project_name=project_name,
        task_name=new_name,
        continue_last_task=True,
        *args,
        **kwargs,
    )

    return task


def connect_whole(
    cfg: DictConfig,
    task: Task,
    name_hyperparams_summary: str = "summary config",
    name_connect_cfg: str = "whole cfg",
):
    """Main connection between ClearML task and hyperparameters tuning.

    Parameters
    ----------
    cfg : DictConfig
        the config that we want to train on
    connect_cfg: Bool
        If yes we connect the
    task : Task
        ClearML task which is the connection between the config and clearml
    name_hyperparams_summary : str, optional
        name of hyperparams  that will be logs in artifact, by default "summary config"
    name_connect_cfg : str, optional
        name of some value that will be displayed in hyperparameters, by default "whole cfg"
    Return
    -------------------
    True to notify the presence of ClearML
    """
    # Connect the hydra Config with Clear ML
    connect_cfg(cfg=cfg, task=task, name=name_connect_cfg)

    # Add some hyperparams in the configuration windows
    connect_hyperparams_summary(
        cfg=cfg, task=task, name=name_hyperparams_summary
    )  # Connect a summary of the config

    # Add information in the configuration/user properties/ properties windows
    set_user_properties(cfg=cfg, task=task)  # Connect some other properties
    return True


def cleamllog_best_ckpt(best_ckpt, task, name_user_properties="path_best_ckpt"):
    if isinstance(task, type(None)):
        pass
    task.set_user_properties(
        {
            "name": name_user_properties,
            "description": "name of the best  ckpt due to training",
            "value": best_ckpt,
        }
    )
    return
