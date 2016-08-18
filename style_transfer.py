# -*- coding: utf-8 -*-
""" nitarshan/style-transfer/style_transfer

Keras (and Theano) implementation of the referenced paper.
Also includes a total variation loss as used in other implementations.
Additionally adds the option to generate output with the colour of the content reference.

# References
- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

"""
from __future__ import division

import numpy as np
import time
import vgg

from argparse import ArgumentParser
from keras import backend as K
from keras.preprocessing import image
from scipy.misc import imread, toimage, fromimage
from scipy.optimize import fmin_l_bfgs_b


""" Command-line arguments """
parser = ArgumentParser(description="Description")

# Inputs and outputs
parser.add_argument('content_path', type=str, help="Path to the content image")
parser.add_argument('style_path', type=str, help="Path to the style image")
#parser.add_argument('-op','--output_prefix', type=str)

# Model parameters
parser.add_argument('-vd', '--vgg_depth', type=int, choices = {16,19}, default=19,
                    help="Depth of VGG model (default: 19)")
parser.add_argument('-vp', "--vgg_pool", type=str, choices={"max","avg"}, default="max",
                    help="Type of pooling layer to use (default: max)")

# Loss weights
parser.add_argument('-cw', '--content_weight', type=float, default=1)
parser.add_argument('-sw', '--style_weight', type=float, default=1e4)
parser.add_argument('-tvw', '--tv_weight', type=float, default=1e2)

# Iteration parameters
parser.add_argument('-i', '--initial', type=str, choices={"noise","content"}, default="content")
parser.add_argument('-it', '--iterations', type=int, default=100)
parser.add_argument('-ch', '--checkpoint', type=int, default=5)
#parser.add_argument('-tol', '--tolerance', type=float, default=0.00001, help='Early stopping (default: 0.00001)')

# Image manipulation
#parser.add_argument('-rs', '--resize_style', type=bool, default=True) # same width as content
#parser.add_argument('-ss', '--style_scale', type=float, default=1.0)
#parser.add_argument('-os', '--output_scale', type=float, default=1.0)
parser.add_argument('-c', '--colour', type=str, choices={"content","generated","both"}, default="generated",
                    help="Which colour scheme to generate output in (default: generated)")

args = parser.parse_args()


""" Helper functions """
def l2_loss(a,b):
    return K.sum(K.square(a-b))

def gram(layer):
    # Obtain the "texture" of a style layer using its gram matrix
    # layer.shape == (batch, channels, height, width)
    # flat.shape == (channels, height*width)
    # gram.shape == (channels, channels)
    layer = K.squeeze(layer, 0) # Remove batch dimension
    flat = K.batch_flatten(layer) # Preserve channel dimension, combine height and width dimensions
    return K.dot(flat, K.transpose(flat)) # gram matrix

def setup_vgg_input(image_path):
    input_image = image.load_img(image_path)
    vgg_input = image.img_to_array(input_image)
    vgg_input = np.expand_dims(vgg_input, axis=0) # Add batch dimension
    vgg_input = vgg.preprocess(vgg_input)
    return vgg_input

def setup_image(vgg_input):
    vgg_input = vgg.deprocess(vgg_input)
    vgg_input = vgg_input.squeeze()
    input_image = image.array_to_img(vgg_input)
    return input_image

def keep_original_colour(orginal_image_path):
    CONTENT_CBCR = imread(args.content_path, mode='YCbCr')
    CONTENT_CBCR[:,:,0] = 0 # Remove Y (grayscale) component

    # Closure with access to CONTENT_CBCR
    def merge_colour(generated_image):
        generated_y = fromimage(generated_image, mode='YCbCr')
        generated_y[:,:,1] = 0 # Remove Cb component
        generated_y[:,:,2] = 0 # Remove Cr component
        generated_array = generated_y + CONTENT_CBCR 
        generated_image = toimage(generated_array, mode='YCbCr')
        return generated_image

    return merge_colour


""" Runtime constants """
CONTENT_WEIGHT = args.content_weight
STYLE_WEIGHT = args.style_weight
TV_WEIGHT = args.tv_weight # Total variation

VGG_DEPTH = args.vgg_depth
VGG_POOL = args.vgg_pool

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1']
if VGG_DEPTH == 19:
    STYLE_LAYERS.append('conv5_1')

ITERATIONS = args.iterations
CHECKPOINT = args.checkpoint
INITIAL_IMAGE = args.initial

COLOUR = args.colour

CONTENT_ARRAY = setup_vgg_input(args.content_path)
STYLE_ARRAY = setup_vgg_input(args.style_path)
merge_colours = keep_original_colour(args.content_path)

IMAGE_HEIGHT = CONTENT_ARRAY.shape[2]
IMAGE_WIDTH = CONTENT_ARRAY.shape[3]
IMAGE_SIZE = 3 * IMAGE_HEIGHT * IMAGE_WIDTH


""" Model setup """
print "Loading VGG-{} model with {}-pooling".format(VGG_DEPTH, VGG_POOL)

model = vgg.build_model(VGG_DEPTH,VGG_POOL)
layers = {layer.name: layer.output for layer in model.layers}

print "Computing reference content"
content_feature_fn = K.function([model.input], [layers[CONTENT_LAYER]])
reference_content = content_feature_fn([CONTENT_ARRAY])[0]

print "Computing reference styles"
style_features_fn = K.function([model.input], [layer.output for layer in model.layers if layer.name in STYLE_LAYERS])
reference_styles = dict(zip(STYLE_LAYERS,style_features_fn([STYLE_ARRAY])))


""" Construct loss graph """
print "Setting up loss graph"

content_loss = l2_loss(layers[CONTENT_LAYER], reference_content) / 2

style_loss = K.variable(0.0)
for layer in STYLE_LAYERS:
    G = gram(layers[layer])
    A = gram(reference_styles[layer])
    style_loss += l2_loss(G,A) / (4*sum(reference_styles[layer].shape)**2)
style_loss = style_loss / len(STYLE_LAYERS) # Weighting factor

vertical_diff = K.abs(model.input[:,:,:IMAGE_HEIGHT-1,:IMAGE_WIDTH-1] - model.input[:,:,1:,:IMAGE_WIDTH-1])
horizontal_diff = K.abs(model.input[:,:,:IMAGE_HEIGHT-1,:IMAGE_WIDTH-1] - model.input[:,:,:IMAGE_HEIGHT-1,1:])
tv_loss = K.sum(K.square(vertical_diff+horizontal_diff))

loss_tensor = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss

gradients_tensor = K.gradients(loss_tensor, model.input)

loss_gradients_fn = K.function([model.input], [loss_tensor, gradients_tensor])
component_losses_fn = K.function([model.input], [content_loss, style_loss, tv_loss])

def loss_gradients_wrapper (vgg_input):
    vgg_input = vgg_input.reshape((1,3,IMAGE_HEIGHT,IMAGE_WIDTH))
    loss, grads = loss_gradients_fn([vgg_input])
    return loss, grads.flatten().astype('float64') # Cryptic FORTRAN errors if not casted to float64


""" Iteration to find optimal image """
print "Begininning iteration"

# Set the initial image for iteration on
if INITIAL_IMAGE == "content":
    vgg_input = CONTENT_ARRAY
elif INITIAL_IMAGE == "noise":
    vgg_input = vgg.preprocess(np.random.uniform(0,255,(1,3,IMAGE_HEIGHT,IMAGE_WIDTH)))

for iteration in xrange(ITERATIONS):
    start_time = time.time()
    vgg_input, loss_val = fmin_l_bfgs_b(loss_gradients_wrapper, vgg_input, maxfun=20)[0:2] # Optimize input to reduce loss

    if (iteration % CHECKPOINT == 0) or (iteration == ITERATIONS-1):
        generated_array = np.reshape(np.copy(vgg_input), (1,3,IMAGE_HEIGHT,IMAGE_WIDTH))
        cont, sty, tv = component_losses_fn([generated_array])
        print("Content: {}, Style: {}, TV: {}".format(cont,sty,tv))
        print("Magnitudes of Content: {}, Style: {}, TV: {}".format(len(str(int(cont))),len(str(int(sty))),len(str(int(tv)))))

        generated_image = setup_image(generated_array)
        if COLOUR in {"generated","both"}:
            generated_image.show()
        if COLOUR in {"content","both"}:
            generated_image = merge_colours(generated_image)
            generated_image.show()

    duration = time.time() - start_time
    print("Iteration {}: Loss value is {}, duration is {} seconds".format(iteration,loss_val,duration))