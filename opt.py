import configargparse

def input_idx(s):
    try:
        a, b, c, d = map(int, s.split(','))
        return [a, b, c, d]
    except:
        raise argparse.ArgumentTypeError("input_idx must be a,b,c,d")

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--training_data_path", type=str, default='data/training/lytro/*.png')
    parser.add_argument("--testing_data_path", type=str, default='data/testing/30scenes/*.png')

    parser.add_argument("--encoder", type=str, default='hrnet')
    parser.add_argument("--inf_in_channels", type=int, default=12)
    parser.add_argument("--rec_in_channels", type=int, default=15)
    parser.add_argument("--rep_channels", type=int, default=64)
    parser.add_argument("--z_channels", type=int, default=64)
    parser.add_argument("--color_channels", type=int, default=3)
    parser.add_argument("--color", type=str, default='rgb')

    parser.add_argument("--input_idx", type=input_idx)
    parser.add_argument("--lf_start_idx", type=int, default=3)
    parser.add_argument("--lf_end_idx", type=int, default=11)
    parser.add_argument("--state_path", type=str)
    parser.add_argument("--num_out_views", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=24)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
