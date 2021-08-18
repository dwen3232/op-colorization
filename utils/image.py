import tensorflow as tf
from matplotlib import pyplot as plt


def generate_images(model, test_input, target):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def plot_dataset_images(dataset):
    plt.figure()
    for input_image, output_image in dataset.take(1):
        # plot input
        plt.subplot(1, 2, 1)
        plt.title('Input Image')
        plt.imshow(input_image[0] * 0.5 + 0.5)
        plt.axis('off')

        # plot output
        plt.subplot(1, 2, 2)
        plt.title('Output Image')
        plt.imshow(output_image[0] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def plot_image(image, title, cmap=None):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

# def load(self, image_file):
#     color = tf.io.read_file(image_file)
#     color = tf.io.decode_png(color, channels=3)
#     color = tf.cast(color, tf.float32)
#
#     gray = tf.io.read_file(image_file.replace('.png', '-gray.png'))
#     gray = tf.io.decode_png(gray, channels=3)
#     gray = tf.cast(gray, tf.float32)
#
#     return gray, color
