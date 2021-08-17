from data_loader.color_data_loader import ColorDataLoader
from models.pix2pix_model import Pix2pixModel
from trainers.pix2pix_trainer import Pix2pixTrainer

from utils.args import get_args
from utils.config import process_config

import tensorflow as tf


def main():
    """
    Classes are hard coded for now. When I add the cycleGAN and ACLGAN
    """

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print('Create the data generator.')
    data_loader = ColorDataLoader(config)
    data_loader.build_datasets()

    print('Create the model.')
    model = Pix2pixModel(config)
    model.build_model()

    print('Create the trainer.')
    trainer = Pix2pixTrainer(model, data_loader, config)
    trainer.load_latest()

    print('Start training the model.')
    trainer.train()

    print('Saving model.')
    model.save()


if __name__ == '__main__':
    main()
