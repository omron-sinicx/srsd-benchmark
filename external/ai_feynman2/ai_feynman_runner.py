import argparse
import os
from pathlib import Path

from aifeynman import run_aifeynman


def get_argparser():
    parser = argparse.ArgumentParser(description='AI Feynman runner')
    parser.add_argument('--src', required=True, help='source file path')
    parser.add_argument('--bftt', type=int, default=60, help='max time for trying combinations of basic operations')
    parser.add_argument('--op', default='14ops.txt', help='op file name')
    parser.add_argument('--poly_deg', type=int, default=3, help='polynomial degree')
    parser.add_argument('--epoch', type=int, default=500, help='op file name')
    return parser


def main(args):
    print(args)
    src_file_path = args.src
    src_dir_path = str(Path(src_file_path).parent) + '/'
    src_file_name = os.path.basename(src_file_path)
    # using test_percentage=0 as we prepare test split separately
    run_aifeynman(src_dir_path, src_file_name, args.bftt, args.op, polyfit_deg=args.poly_deg, NN_epochs=args.epoch, test_percentage=0)


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
