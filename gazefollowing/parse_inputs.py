import argparse

def parse_inputs():
    p = argparse.ArgumentParser()
    p.add_argument('--train_root_dir', type=str,
                        help='path to training set',
                        default=None)
    p.add_argument('--train_mat_file', type=str,
                        help='train annotations',
                        default=None)
    p.add_argument('--test_root_dir', type=str,
                        help='ppath to test set',
                        default=None)
    p.add_argument('--test_mat_file', type=str,
                        help='test annotations',
                        default=None)
    p.add_argument('--batch_size', type=int,
                       help='batch size',
                       default=32)
    p.add_argument('--resume_training', type=bool,
                       help='train from a checkpoint specified by --resume_path',
                       default=False)
    p.add_argument('--resume_path', type=str,
                        help='load model file',
                        default=None)
    p.add_argument('--resume_epoch', type=int,
                       help='epoch to resume',
                       default=25)

    args = p.parse_args()

    return args