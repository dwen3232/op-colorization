import argparse


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = parser.parse_args()
    return args
