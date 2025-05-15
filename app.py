import time, argparse, cv2, torch
import numpy as np
from module import InferMain

import waggle.plugin as plugin
from waggle.data.vision import Camera

TOPIC_CLOUDCOVER = "env.coverage.cloud"
TOPIC_WHERE = "info.cloudcover.where"

plugin.init()


def run(args):
    infermain = InferMain()
    timestamp = time.time()
    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval

    # print("Cloud cover estimation starts...")
    camera = Camera(args.stream)
    while True:
        sample = camera.snapshot()
        image = sample.data
        imagetimestamp = sample.timestamp
        #image = cv2.imread('image.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #timestamp = time.time()

        if args.debug:
            s = time.time()
        ratio, fpoints, data = infermain.run(image)
        if args.debug:
            e = time.time()
            print(f'Time elapsed for inferencing: {e-s} seconds')
            plugin.publish(TOPIC_CLOUDCOVER, f'Time elapsed for inference {e-s} seconds', timestamp=e)

        plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=timestamp)
        plugin.publish(TOPIC_WHERE, fpoint, timestamp=timestamp)
        print(f"Cloud coverage: {ratio} at {fpoints} at time: {imagetimestamp}")
        cv2.imwrite('cloudresult.jpg', data)
        print('saved')
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
    run(parser.parse_args())
