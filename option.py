import argparse
import os

class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='Matching Network')
        parser.add_argument('--dataroot', type=str, default='./datasets',
                            help='path to dataset')
        parser.add_argument('--log-dir', default='./logs',
                            help='folder to output model checkpoints')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()
