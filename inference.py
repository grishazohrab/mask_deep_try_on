import argparse
import copy
import os
from typing import Callable

from tqdm import tqdm

from datasets import create_dataset
from datasets.data_utils import compress_and_save_cloth, remove_extension
from models import create_model
from options.base_options import load
from options.test_options import TestOptions
from util import html
from util.util import PromptOnce
from util.visualizer import save_images


def _setup(subfolder_name, create_webpage=True):
    """
    Setup outdir, create a webpage
    Args:
        subfolder_name: name of the outdir and where the webpage files should go

    Returns:

    """
    out_dir = get_out_dir(subfolder_name)
    PromptOnce.makedirs(out_dir, not opt.no_confirm)
    webpage = None
    if create_webpage:
        webpage = html.HTML(
            out_dir,
            f"Experiment = {opt.name}, Phase = {subfolder_name} inference, "
            f"Loaded Epoch = {opt.load_epoch}",
        )
    return out_dir, webpage


def get_out_dir(subfolder_name):
    return os.path.join(opt.results_dir, subfolder_name)


def _rebuild_from_checkpoint(checkpoint_file, same_crop_load_size=False, **ds_kwargs):
    """
    Loads a model and dataset based on the config in a particular dir.
    Args:
        checkpoint_file: dir containing args.json and model checkpoints
        **ds_kwargs: override kwargs for dataset

    Returns: loaded model, initialized dataset

    """
    checkpoint_dir = os.path.dirname(checkpoint_file)
    # read the config file  so we can load in the model
    loaded_opt = load(copy.deepcopy(opt), os.path.join(checkpoint_dir, "args.json"))
    # force certain attributes in the loaded cfg
    override_namespace(
        loaded_opt,
        is_train=False,
        batch_size=1,
        shuffle_data=opt.shuffle_data,  # let inference opt take precedence
    )
    if same_crop_load_size:  # need to override this if we're using intermediates
        loaded_opt.load_size = loaded_opt.crop_size
    model = create_model(loaded_opt)
    # loads the checkpoint
    model.load_model_weights("generator", checkpoint_file).eval()
    model.print_networks(opt.verbose)

    dataset = create_dataset(loaded_opt, **ds_kwargs)

    return model, dataset


def override_namespace(namespace, **kwargs):
    """
    Simply overrides the attributes in the object with the specified keyword arguments
    Args:
        namespace: argparse.Namespace object
        **kwargs: keyword/value pairs to use as override
    """
    assert isinstance(namespace, argparse.Namespace)
    for k, v in kwargs.items():
        setattr(namespace, k, v)


def _run_test_loop(model, dataset, webpage=None, iteration_post_hook: Callable = None):
    """

    Args:
        model: object that extends BaseModel
        dataset: object that extends BaseDataset
        webpage: webpage object for saving
        iteration_post_hook: a function to call at the end of every iteration

    Returns:

    """

    total = min(len(dataset), opt.max_dataset_size)
    with tqdm(total=total, unit="img") as pbar:
        for i, data in enumerate(dataset):
            if i >= total:
                break
            model.set_input(data)  # set input
            model.test()  # forward pass
            image_paths = model.get_image_paths()  # ids of the loaded images

            if webpage:
                visuals = model.get_current_visuals()
                save_images(webpage, visuals, image_paths, width=opt.display_winsize)

            if iteration_post_hook:
                iteration_post_hook(local=locals())

            pbar.update()

    if webpage:
        webpage.save()


if __name__ == "__main__":
    config = TestOptions()
    config.parse()
    opt = config.opt

    # override checkpoint options
    if opt.checkpoint:
        # TODO: run test
        pass
    print("\nDone!")
