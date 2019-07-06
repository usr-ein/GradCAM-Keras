import argparse
from path_type import PathType


def parse_args():
    parser = argparse.ArgumentParser(description='Explains what VGG19 looks at to classify the given image.')

    parser.add_argument('--input', '-i',
                        metavar="/path/to/input/image.png",
                        type=PathType(exists=True, type='file'),
                        required=True,
                        help="Image to run onto, can be any format.")
    parser.add_argument('--output', '-o',
                        metavar="/path/to/output/graph.png",
                        type=PathType(exists=None, type='file'),
                        required=False,
                        help="Where to generate the output visualisation.")
    parser.add_argument('--layer', '-l',
                        choices=['block1_conv1',
                                 'block1_conv2',
                                 'block2_conv1',
                                 'block2_conv2',
                                 'block3_conv1',
                                 'block3_conv2',
                                 'block3_conv3',
                                 'block3_conv4',
                                 'block4_conv1',
                                 'block4_conv2',
                                 'block4_conv3',
                                 'block4_conv4',
                                 'block5_conv1',
                                 'block5_conv2',
                                 'block5_conv3',
                                 'block5_conv4'],
                        required=True,
                        help="Layer at which to \"look at\".")
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="If set, doesn't show the plot and only saves the output (if one is provided).")

    return parser.parse_args()
