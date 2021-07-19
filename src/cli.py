import argparse
import os
import numpy
import torch
from torchvision import transforms
import train
import eval as evaluate

from util import get_config, get_device

parser = argparse.ArgumentParser(description='Reinforcement Learning environments')
parser.add_argument('--ckpf', help="path to model checkpoint file (to continue training)")
parser.add_argument('--train', action='store_true', help='')
parser.add_argument('--evaluate', action='store_true', help='evaluate a [pre]trained model')
parser.add_argument('--image_path', help='')


args = parser.parse_args()
config = get_config()
device = get_device()
tfs = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

if args.train:
    train(config)

elif args.evaluate:
    evaluate(args.ckpf, input)