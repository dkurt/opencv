import argparse
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(
        description='Run pix2pix models using OpenCV. '
                    'Visit Torch implementation repository (https://github.com/phillipi/pix2pix) and '
                    'TensorFlow port repository (https://github.com/affinelayer/pix2pix-tensorflow) for details.')
parser.add_argument('--model', help='Path to TensorFlow model', required=True)
parser.add_argument('--input', help='Path to input image. RGB images are converted to '
                                    'edges using Canny edge detector. Single channel '
                                    'grayscale images considered to be edges. Skip this '
                                    'argument to enable drawing mode.')
args = parser.parse_args()

net = cv.dnn.readNetFromTensorflow(args.model)

inpWidth = 256
inpHeight = 256

def process(edges):
    blob = cv.dnn.blobFromImage(edges, 1.0 / 127.5, (inpWidth, inpHeight), (127.5, 127.5, 127.5), True, False)
    net.setInput(blob)
    out = net.forward()
    out *= 127.5
    out += 127.5
    return out.transpose(0, 2, 3, 1).reshape(inpHeight, inpWidth, 3)[:,:,[2,1,0]]


if args.input:
    inp = cv.imread(args.input, cv.IMREAD_UNCHANGED)
    if len(inp.shape) == 3:
        inp = cv.Canny(inp, 100, 200)
        inp = 255 - inp
        inp = np.concatenate((np.expand_dims(inp, -1),) * 3, axis=-1)

    out = process(inp)
    out = cv.resize(out, (inp.shape[1], inp.shape[0]))

    # Read image again to have an RGB array.
    img = cv.imread(args.input)
    cv.namedWindow('pix2pix using OpenCV', cv.WINDOW_NORMAL)
    cv.imshow('pix2pix using OpenCV', np.concatenate((img, inp, out.astype(np.uint8)), axis=1))
    cv.waitKey()
