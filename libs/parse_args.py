import argparse, warnings, datetime, os


def parse_args(args):
    """ Parse the arguments.
        """
    parser = argparse.ArgumentParser(description='Simple training script for training an autoencoder for SPV clustering')
    # subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    # subparsers.required = True

    # def csv_list(string):
    #     return string.split(',')

    # sail_parser = subparsers.add_parser('sail')
    # sail_parser.add_argument('annotations',         help='Path to pickle-file containing annotations for training.')
    # sail_parser.add_argument('classes',             help='Path to a CSV file containing class label mapping.')
    # sail_parser.add_argument('--train-data-base-path', help='Path to the directory with the train data itself', required=True)
    # sail_parser.add_argument('--train-masks-base-path', help='Path to the directory with the masks for train data', required=True)
    # sail_parser.add_argument('--val-annotations',   help='Path to pickle-file containing annotations for validation (optional).')
    # sail_parser.add_argument('--val-data-base-path', help='Path to the directory with the validation data itself')
    # sail_parser.add_argument('--val-masks-base-path', help='Path to the directory with the masks for validation data')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',                help='Resume training from a snapshot.')
    # group.add_argument('--imagenet-weights',        help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
    #                    action='store_const', const=True, default=True)
    # group.add_argument('--weights',                 help='Initialize the model with weights from a file.')
    # group.add_argument('--no-weights',              help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--run-name',               help='name for the current run (directories will be created based on this name)', default='devel')
    # parser.add_argument('--backbone',               help='Backbone model used by retinanet.',
    #                     default='resnet50', type=str)
    parser.add_argument('--batch-size',             help='Size of the batches.',
                        default=32, type=int)
    parser.add_argument('--val-batch-size',         help='Size of the batches for evaluation.',
                        default=32, type=int)
    parser.add_argument('--gpu',                    help='Id of the GPU to use (as reported by nvidia-smi).')
    # parser.add_argument('--multi-gpu',              help='Number of GPUs to use for parallel processing.',
    #                     type=int, default=0)
    # parser.add_argument('--multi-gpu-force',        help='Extra flag needed to enable (experimental) multi-gpu support.',
    #                     action='store_true')
    parser.add_argument('--epochs',                 help='Number of epochs to train.',
                        type=int, default=200)
    parser.add_argument('--steps-per-epoch',                  help='Number of steps per epoch.',
                        type=int)
    # parser.add_argument('--steps',                  help='Number of steps per epoch.', type=int)
    parser.add_argument('--val-steps',              help='Number of steps per validation run.',
                        type=int, default=100)
    # parser.add_argument('--val-steps',              help='Number of steps per validation run.', type=int)
    # parser.add_argument('--lr',                     help='Initial learning rate.',
    #                     type=float, default=1e-5)
    # parser.add_argument('--snapshot-path',          help='Path to store snapshots of models during training (defaults to \'./snapshots\')',
    #                     default='./snapshots')
    # parser.add_argument('--tensorboard-dir',        help='Log directory for Tensorboard output',
    #                     default='./logs')
    # parser.add_argument('--logs-path',              help='Log directory for text output',
    #                     default='./logs')
    parser.add_argument('--no-snapshots',           help='Disable saving snapshots.',
                        dest='snapshots', action='store_false')

    parser.add_argument('--variational',            help='make the autoencoder variational',
                        dest='variational', action='store_true')
    parser.add_argument('--debug', help='launch in DEBUG mode',
                        dest='debug', action='store_true')
    parser.add_argument('--embeddims', help='Embeddings dimensionality',
                        type=int, default=128)
    # parser.add_argument('--no-evaluation',          help='Disable per epoch evaluation.',
    #                     dest='evaluation', action='store_false')
    # parser.add_argument('--freeze-backbone',        help='Freeze training of backbone layers.',
    #                     action='store_true')
    # parser.add_argument('--random-transform',       help='Randomly transform image and annotations.',
    #                     action='store_true')
    # parser.add_argument('--image-min-side',         help='Rescale the image so the smallest side is min_side.',
    #                     type=int, default=800)
    # parser.add_argument('--image-max-side',         help='Rescale the image if the largest side is larger than max_side.',
    #                     type=int, default=1333)
    # parser.add_argument('--config',                 help='Path to a configuration parameters .ini file.')
    # parser.add_argument('--weighted-average',       help='Compute the mAP using the weighted average of precisions among classes.',
    #                     action='store_true')

    # parser.add_argument('--lr-scheduler',           help='Flag enabling Cosine annealing learning rate scheduler with periodic restarts',
    #                     action='store_true', default=False)
    # parser.add_argument('--reduce-lr-on-plateau',   help='Flag enabling ReduceLROnPlateau callback',
    #                     action='store_true', default=False)

    # Fit generator arguments
    # parser.add_argument('--workers',                help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0',
    #                     type=int, default=1)
    # parser.add_argument('--max-queue-size',         help='Queue length for multiprocessing workers in fit generator.',
    #                     type=int, default=10)

    return preprocess_args(parser.parse_args(args))


def preprocess_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
        For example, batch_size < num_gpus
        Intended to raise errors prior to backend initialisation.

        Args
            parsed_args: parser.parse_args()

        Returns
            parsed_args
        """

    # if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
    #     raise ValueError(
    #         "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
    #                                                                                          parsed_args.multi_gpu))

    # if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
    #     raise ValueError(
    #         "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
    #                                                                                             parsed_args.snapshot))

    # if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
    #     raise ValueError(
    #         "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    # if 'resnet' not in parsed_args.backbone:
    #     warnings.warn(
    #         'Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    parsed_args.snapshot_path = os.path.join('./snapshots', parsed_args.run_name)
    parsed_args.tensorboard_dir = os.path.join('./logs', parsed_args.run_name)
    parsed_args.logs_path = os.path.join('./logs', parsed_args.run_name)

    return parsed_args