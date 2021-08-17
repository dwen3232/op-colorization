from base.base_model import BaseModel

import tensorflow as tf
from dotmap import DotMap
from pathlib import Path


class Pix2pixModel(BaseModel):
    def __init__(self, config=DotMap()):
        super().__init__(config)
        self.exp_name = config.exp.get('name', 'pix2pix')
        self.img_height = self.config.exp.get('img_height', 256)
        self.img_width = self.config.exp.get('img_width', 256)

        self.plot_dir = Path(self.config.model.get('plot_dir', './plots')) / self.exp_name
        self.weights_dir = Path(self.config.model.get('weights_dir', './weights')) / self.exp_name


        # probably shouldn't be changed
        self.LAMBDA = 100

        self.generator = None
        self.discriminator = None

    def save(self):
        if self.generator is None or self.discriminator is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.generator.save_weights(str(self.weights_dir / 'generator'))
        self.discriminator.save_weights(str(self.weights_dir / 'discriminator'))
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self):
        if self.generator is None or self.discriminator is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(self.weights_dir))
        self.generator.load_weights(str(self.weights_dir / 'generator'))
        self.discriminator.load_weights(str(self.weights_dir / 'generator'))
        print("Model loaded")

    def build_model(self):
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def plot_model(self):
        if self.generator is None or self.discriminator is None:
            raise Exception("You have to build the model first.")

        tf.keras.utils.plot_model(self.generator, to_file=str(self.plot_dir / 'generator.png'),
                                  show_shapes=True, dpi=64)
        tf.keras.utils.plot_model(self.discriminator, to_file=str(self.plot_dir / 'discriminator.png'),
                                  show_shapes=True, dpi=64)

    def build_generator(self):
        # NOTE: move this somewhere else?
        output_channels = 3

        # NOTE: this shape might not be valid
        inputs = tf.keras.layers.Input(shape=[self.img_height, self.img_width, 1])

        down_stack = [
            down_sample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            down_sample(128, 4),  # (batch_size, 64, 64, 128)
            down_sample(256, 4),  # (batch_size, 32, 32, 256)
            down_sample(512, 4),  # (batch_size, 16, 16, 512)
            down_sample(512, 4),  # (batch_size, 8, 8, 512)
            down_sample(512, 4),  # (batch_size, 4, 4, 512)
            down_sample(512, 4),  # (batch_size, 2, 2, 512)
            down_sample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            up_sample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            up_sample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            up_sample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            up_sample(512, 4),  # (batch_size, 16, 16, 1024)
            up_sample(256, 4),  # (batch_size, 32, 32, 512)
            up_sample(128, 4),  # (batch_size, 64, 64, 256)
            up_sample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Down sampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Up sampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def build_discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        # NOTE: this shape might not be valid
        inp = tf.keras.layers.Input(shape=[self.img_height, self.img_width, 1], name='input_image')
        tar = tf.keras.layers.Input(shape=[self.img_height, self.img_width, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = down_sample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = down_sample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = down_sample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)


    def generator_loss(self, disc_generated_output, gen_output, target):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

        return total_gen_loss, gan_loss, l1_loss


    def discriminator_loss(self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss


def down_sample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def up_sample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result



