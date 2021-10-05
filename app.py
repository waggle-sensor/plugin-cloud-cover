import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main

import waggle.plugin as plugin
from waggle.data.vision import Camera

TOPIC_CLOUDCOVER = "env.coverage.cloud"

plugin.init()


def run(args):
    unet_main = Unet_Main()
    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval

    # print("Cloud cover estimation starts...")
    camera = Camera(args.stream)
    # camera = Camera()
    while True:
        sample = camera.snapshot()
        image = sample.data
        timestamp = sample.timestamp
        # image = cv2.imread('test4.jpg')
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # timestamp = time.time()

        if args.debug:
            s = time.time()
        ratio, hi = unet_main.run(image, out_threshold=args.threshold)
        if args.debug:
            e = time.time()
            print(f'Time elapsed for inferencing: {e-s} seconds')

        plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=timestamp)
        print(f"Cloud coverage: {ratio} at time: {timestamp}")
        # plugin.upload_file(hi, timestamp=timestamp)
        # print(f"Cloud coverage result at time: {timestamp}")

        if sampling_countdown > 0:
            sampling_countdown -= 1
        elif sampling_countdown == 0:
            sample.save('sample.jpg')
            plugin.upload_file('sample.jpg')
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
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-threshold', dest='threshold',
        action='store', default=0.9, type=float,
        help='Cloud pixel determination threshold')
    run(parser.parse_args())
