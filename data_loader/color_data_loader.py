from base.base_data_loader import BaseDataLoader
import tensorflow as tf

from pathlib import Path
from dotmap import DotMap


class ColorDataLoader(BaseDataLoader):
    """
    Creates training and testing datasets from loaded data.
    Datasets are input and output image pairs, where the  input image is the grayscale of the output image.
    Input is of shape (h, w, 1)
    Output is of shape (h, w, 3)
    """

    def __init__(self, config=DotMap()):
        super().__init__(config)
        # Loads config parameters with default values
        self.buffer_size = self.config.loader.get('buffer_size', 400)
        self.batch_size = self.config.loader.get('batch_size', 1)

        self.img_height = self.config.exp.get('img_height', 256)
        self.img_width = self.config.exp.get('img_width', 256)

        self.height_jitter = self.config.loader.get('height_jitter', 20)
        self.width_jitter = self.config.loader.get('width_jitter', 20)

        self.data_dir = Path(self.config.loader.get('data_dir', './datasets/color'))
        self.train_percent = self.config.loader.get('train_percent', 0.7)
        self.test_percent = self.config.loader.get('test_percent', 0.3)

        self.dataset = None
        self.train_data = None
        self.test_data = None

    def build_datasets(self):
        """
        Builds three datasets in total:
        1. Dataset of all file names in data_dir
        2. Training dataset of input and output pairs
        3. Testing dataset of input and output pairs
        """
        # creates dataset of shuffled file names
        print('Creating file name dataset...')
        dataset = tf.data.Dataset.list_files(str(self.data_dir / '*.png'))
        dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=False)
        self.dataset = dataset

        dataset_size = dataset.cardinality().numpy()

        print('Creating training dataset...')
        # creates training set from dataset by mapping and batching
        train_size = int(self.train_percent * dataset_size)
        train_data = dataset.take(train_size)
        remaining_data = dataset.skip(train_size)
        train_data = train_data.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_data = train_data.batch(self.batch_size)
        self.train_data = train_data

        print('Creating testing dataset...')
        # creates testing set from dataset by mapping and batching
        test_size = int(self.test_percent * dataset_size)
        test_data = remaining_data.take(test_size)
        test_data = test_data.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_data = test_data.batch(self.batch_size)
        self.test_data = test_data

    def load_image(self, image_file):
        """

        :param image_file: file name to load color image from
        :return: input and output images
        """
        # loads image file
        real_image = load(image_file)
        # adds jitter to color image
        real_image = random_jitter(real_image,
                                   self.img_height, self.img_width,
                                   self.height_jitter, self.width_jitter)
        # creates input_image by converting the color image to grayscale
        input_image = tf.image.rgb_to_grayscale(real_image)
        # normalizes grayscale and color images
        input_image, real_image = normalize(input_image, real_image)

        return input_image, real_image

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


# HELPER METHODS

def load(image_file):
    """
    Loads RGB image from file name

    :param image_file: file name to load image from
    :return: loaded image
    """
    color = tf.io.read_file(image_file)
    color = tf.io.decode_png(color, channels=3)
    color = tf.cast(color, tf.float32)

    return color


def resize(real_image, height, width):
    """
    Resizes input image to given height and width

    :param real_image: image to resize
    :param height: height of returned image
    :param width: width of returned image
    :return: resized image
    """
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.AREA)
    return real_image


def random_crop(real_image, height, width):
    """
    Crops input image to given height and width

    :param real_image: image to crop
    :param height: height of returned image
    :param width: width of returned image
    :return: cropped image
    """
    cropped_image = tf.image.random_crop(
        real_image, size=[height, width, 3])
    return cropped_image


def normalize(input_image, real_image):
    """
    Normalizes input and real images such that all pixel values are in [-1, 1]

    :param input_image: grayscale input image
    :param real_image: color output image
    :return: normalized input_image and real_image
    """
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(real_image, height, width, height_jitter, width_jitter):
    """
    Adds random jitter to input image

    :param real_image: image to add jitter to
    :param height: height of returned image
    :param width: width of returned image
    :param height_jitter: maximum jitter height-wise
    :param width_jitter: maximum jitter width-wise
    :return: image with random jitter added
    """
    # resizes real_image to (height+height_jitter, width+width_jitter, 3)
    real_image = resize(real_image, height + height_jitter, width + width_jitter)
    # crops real_image to (height, width, 3)
    real_image = random_crop(real_image, height, width)

    if tf.random.uniform(()) > 0.5:
        real_image = tf.image.flip_left_right(real_image)

    return real_image
