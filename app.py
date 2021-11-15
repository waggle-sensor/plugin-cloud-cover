import time
import argparse

import cv2
import torch
import numpy as np

from unet_module import Unet_Main
from unet_module import dewarping

import waggle.plugin as plugin
from waggle.data.vision import Camera

TOPIC_CLOUDCOVER = "env.coverage.cloud"

plugin.init()


def run(args):
    timestamp = time.time()
    plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Start', timestamp=timestamp)
    print(f"Starts at time: {timestamp}")
    timestamp = time.time()
    plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Loading Model', timestamp=timestamp)
    print(f"Loading Model at time: {timestamp}")
    unet_main = Unet_Main()
    timestamp = time.time()
    plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Model Loaded', timestamp=timestamp)
    print(f"Model Loaded at time: {timestamp}")
    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval

    plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Loading an Image', timestamp=timestamp)
    print(f"Loading an Image at time: {timestamp}")
    # print("Cloud cover estimation starts...")
    camera = Camera(args.stream)
    #camera = Camera()
    while True:
        sample = camera.snapshot()
        image = sample.data
        imagetimestamp = sample.timestamp
        #image = cv2.imread('test.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #imagetimestamp = time.time()
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: an Image Loaded', timestamp=timestamp)
        print(f"Image Loaded at time: {timestamp}")

        if args.debug:
            s = time.time()
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Starting Preprocessing', timestamp=timestamp)
        print(f"Starting Preprocessing at time: {timestamp}")
        h, w, c = image.shape
        image = dewarping(image, h, w)
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: End Preprocessing', timestamp=timestamp)
        print(f"End Preprocessing at time: {timestamp}")
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: Starting Inference', timestamp=timestamp)
        print(f"Inference Starts at time: {timestamp}")
        ratio, hi = unet_main.run(image, out_threshold=args.threshold)
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, 'Cloud Cover Estimator: End Inference', timestamp=timestamp)
        print(f"End Inference at time: {timestamp}")
        if args.debug:
            e = time.time()
            print(f'Time elapsed for inferencing: {e-s} seconds')
        timestamp = time.time()
        plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=imagetimestamp)
        print(f"Cloud coverage: {ratio} at time: {imagetimestamp}")
        cv2.imwrite('cloudresult.jpg', hi)
        print('saved')
        exit(0)
        plugin.upload_file('cloudresult.jpg')
        print(f"Cloud coverage result at time: {imagetimestamp}")

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

        exit(0)


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
