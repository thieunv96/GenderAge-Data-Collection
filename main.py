import cv2
import time
from snapper import Snapper
import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network for ")
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    opt = parser.parse_args() 
    configs = yaml.safe_load(open(opt.config))
    print(configs)
    snap = Snapper(configs)
    snap.run()