import json
import struct
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from tensorflow.python.tools import optimize_for_inference_lib

parser = argparse.ArgumentParser(description='This script is used to convert quantized '
                                             'pix2pix models from https://github.com/affinelayer/pix2pix-tensorflow-models '
                                             'into TensorFlow graph.')
parser.add_argument('input', help='Path to input .pict file')
parser.add_argument('output', help='Path to output .pb TensorFlow model')
args = parser.parse_args()

################################################################################
# Read quantized weights
################################################################################
with open(args.input, 'rt') as f:
    data = f.read()

parts = []
offset = 0
while offset < len(data):
    size = struct.unpack('>i', data[offset:offset+4])[0]
    offset += 4
    parts.append(data[offset:offset+size])
    offset += size

shapes = json.loads(parts[0])
assert(len(parts[1]) % 4 == 0)
index = struct.unpack('<%df' % (len(parts[1]) / 4), parts[1])
encoded = struct.unpack('<%dB' % len(parts[2]), parts[2])

qMin = np.min(index)
qMax = np.max(index)

################################################################################
# Quantize float values again using TensorFlow's quantize operation.
################################################################################
with tf.Session(graph=tf.Graph()) as sess:
    inp = tf.placeholder(tf.float32, [len(index)])
    q = tf.quantize(inp, qMin, qMax, tf.quint8, 'MIN_FIRST')

    sess.run(tf.global_variables_initializer())

    out = sess.run(q, feed_dict={inp: index})
    index = out.output

arr = [index[e] for e in encoded]

weights = {}
offset = 0
for entry in shapes:
    shape = entry['shape']
    name = entry['name']
    weights[name] = np.array(arr[offset:offset+np.prod(shape)], dtype=np.uint8).reshape(shape)
    offset += np.prod(shape)

################################################################################
# Build a model
################################################################################
def convLayer(x, w, b):
    w = tf.dequantize(w, qMin, qMax, 'MIN_FIRST')
    b = tf.dequantize(b, qMin, qMax, 'MIN_FIRST')
    conv = tf.nn.conv2d(x, w, strides=(1, 2, 2, 1), padding='SAME')
    return tf.nn.bias_add(conv, b)

def batchNorm(x, gamma, beta):
    gamma = tf.dequantize(gamma, qMin, qMax, 'MIN_FIRST')
    beta = tf.dequantize(beta, qMin, qMax, 'MIN_FIRST')
    return tf.nn.fused_batch_norm(x, gamma, beta, epsilon=1e-5, is_training=True)[0]

def deconvLayer(x, w, b):
    w = tf.dequantize(w, qMin, qMax, 'MIN_FIRST')
    b = tf.dequantize(b, qMin, qMax, 'MIN_FIRST')
    outShape = output_shape=[1, int(x.shape[1])*2, int(x.shape[2])*2, int(w.shape[2])]
    deconv = tf.nn.conv2d_transpose(relu, w, outShape,
                                    strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.bias_add(deconv, b)


edges = tf.placeholder(tf.float32, [1, 256, 256, 3], 'input')

w = tf.Variable(weights['generator/encoder_1/conv2d/kernel'], dtype=tf.quint8)
b = tf.Variable(weights['generator/encoder_1/conv2d/bias'], dtype=tf.quint8)
layers = [convLayer(edges, w, b)]

for i in range(2, 9):
    scope = 'generator/encoder_%d' % i
    w = tf.Variable(weights[scope + '/conv2d/kernel'], dtype=tf.quint8)
    b = tf.Variable(weights[scope + '/conv2d/bias'], dtype=tf.quint8)
    rectified = tf.nn.leaky_relu(layers[-1], alpha=0.2)
    conv = convLayer(rectified, w, b)

    gamma = tf.Variable(weights[scope + '/batch_normalization/gamma'], dtype=tf.quint8)
    beta = tf.Variable(weights[scope + '/batch_normalization/beta'], dtype=tf.quint8)
    layers += [batchNorm(conv, gamma, beta)]

for i in range(8, 1, -1):
    if i == 8:
        inp = layers[-1]
    else:
        inp = tf.concat([layers[-1], layers[i - 1]], axis=3)
    relu = tf.nn.relu(inp)

    scope = 'generator/decoder_%d' % i
    w = tf.Variable(weights[scope + '/conv2d_transpose/kernel'], dtype=tf.quint8)
    b = tf.Variable(weights[scope + '/conv2d_transpose/bias'], dtype=tf.quint8)
    deconv = deconvLayer(relu, w, b)

    gamma = tf.Variable(weights[scope + '/batch_normalization/gamma'], dtype=tf.quint8)
    beta = tf.Variable(weights[scope + '/batch_normalization/beta'], dtype=tf.quint8)
    layers += [batchNorm(deconv, gamma, beta)]

inp = tf.concat([layers[-1], layers[0]], axis=3)
relu = tf.nn.relu(inp)
w = tf.Variable(weights['generator/decoder_1/conv2d_transpose/kernel'], dtype=tf.quint8)
b = tf.Variable(weights['generator/decoder_1/conv2d_transpose/bias'], dtype=tf.quint8)
deconv = deconvLayer(relu, w, b)
out = tf.tanh(deconv, name='output')

################################################################################
# Save a model
################################################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    graph_def = sess.graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
    with tf.gfile.FastGFile(args.output, 'wb') as f:
        f.write(graph_def.SerializeToString())
