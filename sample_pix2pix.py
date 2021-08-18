from models.pix2pix_model import Pix2pixModel
from utils.args import get_args
from utils.config import process_config

import tensorflow as tf

from matplotlib import pyplot as plt


# Quickly written script for testing images. Not very idiomatic in the slightest; will write a new script to
# generalize this to all models after I finish writing the CycleGAN
from utils.image import plot_image


def main():
    try:
        config = process_config('configs/pix2pix.json')
        image_file = '0604-003'
        image_path = f'datasets/color/{image_file}.png'
    except:
        print("missing or invalid arguments")
        exit(0)

    print('Create the model.')
    model = Pix2pixModel(config)
    model.build_model()
    model.load()

    color = tf.io.read_file(image_path)
    color = tf.io.decode_png(color, channels=3)
    color = tf.cast(color, tf.float32)
    color = tf.image.resize(color, [256, 256],
                            method=tf.image.ResizeMethod.AREA)

    gray = tf.image.rgb_to_grayscale(color)

    gen_out = model.generator((gray[tf.newaxis, :, :, :] / 127.5) - 1, training=True)

    plot_image(gray, "Gray Image", "gray")
    plot_image(color/255.0, "Color Image")
    plot_image(gen_out[0] * 0.5 + 0.5, "Generated Image")



if __name__ == '__main__':
    main()
