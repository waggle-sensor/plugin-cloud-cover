import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main

import waggle.plugin as plugin
from waggle.data import open_data_source
from waggle.data.vision import Camera

TOPIC_INPUT_IMAGE = "sky_image"
TOPIC_CLOUDCOVER = "env.coverage.cloud"

plugin.init()
camera = Camera()

def run(args):
    unet_main = Unet_Main()

    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval
    print("Cloud cover estimation starts...")
    while True:
        image = camera.snapshot()
        timestamp = image.timestamp
        #image = cv2.imread('test.jpg')
        #timestamp = time.time()

        if args.debug:
            s = time.time()
        ratio = unet_main.run(image)
        if args.debug:
            e = time.time()
            print(e-s)

        plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=timestamp)
        if args.debug:
            print(f"Cloud coverage: {ratio}")
            print(f"Measures published")

        if sampling_countdown > 0:
            sampling_countdown -= 1
        elif sampling_countdown == 0:
            cv2.imwrite('/tmp/sample.jpg', image)
            plugin.upload_file('/tmp/sample.jpg')
            if args.debug:
                print("A sample is published")
            # Reset the count
            sampling_countdown = args.sampling_interval

        if args.interval > 0:
            time.sleep(args.interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    run(parser.parse_args())
