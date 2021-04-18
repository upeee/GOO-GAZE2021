import argparse

def parse_inputs():

    # Command line arguments
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', type=str,
                        help='path to training set',
                        default=None)
    p.add_argument('--train_annotation', type=str,
                        help='train annotations (pickle/mat)',
                        default=None)
    p.add_argument('--test_dir', type=str,
                        help='path to test set',
                        default=None)
    p.add_argument('--test_annotation', type=str,
                        help='test annotations (pickle/mat)',
                        default=None)
    p.add_argument('--resume_training',
                       help='train from a checkpoint specified by --resume_path',
                       action='store_true')
    p.add_argument('--resume_path', type=str,
                        help='load model file',
                        default=None)

    # Save files and directories
    p.add_argument('--log_file', type=str,
                    help='logging training error',
                    default='train.log')
    p.add_argument('--save_model_dir', type=str,
                    help='directory for saving models',
                    default='./saved_models/temp/')

    # Training hyperparams
    p.add_argument('--baseline', type=str,
                        help='recasens or gazenet',
                        default='gazenet')
    p.add_argument('--gazemask',
                       help='Enable adding gaze object mask to target heatmap. Only on GOOdataset.',
                       action='store_true')
    p.add_argument('--batch_size', type=int,
                       help='batch size',
                       default=32)
    p.add_argument('--init_lr', type=float,
                        help='initial learning rate',
                        default=1e-4)
    p.add_argument('--max_epochs', type=int,
                       help='max epochs to end training',
                       default=25)

    # Chong specific
    p.add_argument("--init_weights", type=str, default="saved_model/chong_init_weights_spatial.pt", help="initial weights")


    args = p.parse_args()

    return args
