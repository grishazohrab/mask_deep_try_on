import argparse

from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self, **defaults):
        super().__init__()
        self.is_train = False
        parser = self._parser

        parser.set_defaults(max_dataset_size=50, shuffle_data=False)
        parser.add_argument(
            "--interval",
            metavar="N",
            default=1,
            type=int,
            help="only run every n images",
        )
        parser.add_argument(
            "--checkpoint",
            help="Shorthand for both warp and texture checkpoint to use the 'latest' "
                 "generator file (or specify using --load_epoch). This should be the "
                 "root dir containing warp/ and texture/ checkpoint folders.",
        )
        parser.add_argument(
            "--results_dir",
            default="results",
            help="folder to output intermediate and final results",
        )
        parser.add_argument(
            "--skip_intermediates",
            action="store_true",
            help="choose not to save intermediate cloth visuals as images for warp "
            "stage (instead, just save .npz files)",
        )

        parser.add_argument(
            "--dataroot",
            required=False,
            help="path to dataroot",
        )
        # remove arguments
        parser.add_argument(
            "--model", help=argparse.SUPPRESS
        )  # remove model as we restore from checkpoint
        parser.add_argument("--name", default="", help=argparse.SUPPRESS)

        parser.set_defaults(**defaults)

    @staticmethod
    def _validate(opt):
        super(TestOptions, TestOptions)._validate(opt)

        if not opt.dataroot:
            raise ValueError("There is not `data root` dir")
