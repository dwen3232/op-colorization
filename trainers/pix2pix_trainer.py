from base.base_trainer import BaseTrainer
from utils.image import generate_images

import time
import datetime
from pathlib import Path
from dotmap import DotMap
import tensorflow as tf


class Pix2pixTrainer(BaseTrainer):
    def __init__(self, model, data, config=DotMap()):
        super().__init__(model, data, config)
        self.exp_name = config.exp.get('name', 'pix2pix')

        self.steps = self.config.trainer.get('steps', 40000)
        self.checkpoint_dir = Path(self.config.trainer.get('checkpoint_dir', './training_checkpoints')) / self.exp_name
        self.log_dir = Path(self.config.trainer.get('log_dir', './logs')) / self.exp_name


        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.model.generator,
                                              discriminator=self.model.discriminator)

        self.summary_writer = tf.summary.create_file_writer(
            str(self.log_dir / 'fit' / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    def load_latest(self):
        latest = tf.train.latest_checkpoint(str(self.checkpoint_dir))
        print(f'Loading latest from {latest}')
        self.checkpoint.restore(latest)

    def train(self):
        self.fit(self.data.get_train_data(), self.data.get_test_data(), steps=self.steps)

    def fit(self, train_ds, test_ds, steps):

        start = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if step % 1000 == 0:
                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

                start = time.time()

                print(f"Step: {step // 1000}k")

            self.train_step(input_image, target, step)

            # Training step
            if (step + 1) % 10 == 0:
                print('.', end='', flush=True)

            # Save (checkpoint) the model every 5k steps at ./[checkpoint_dir]/[exp_name]/ckpt
            if (step + 1) % 5000 == 0:
                self.checkpoint.save(file_prefix=str(self.checkpoint_dir / "ckpt"))

        for i in range(10):
            example_input, example_target = next(iter(test_ds.take(1)))
            generate_images(self.model.generator, example_input, example_target)


    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.model.generator(input_image, training=True)

            disc_real_output = self.model.discriminator([input_image, target], training=True)
            disc_generated_output = self.model.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.model.generator_loss(disc_generated_output, gen_output,
                                                                                  target)
            disc_loss = self.model.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.model.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.model.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                 self.model.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                     self.model.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)

