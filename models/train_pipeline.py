import yaml
import numpy as np
import sys
sys.path.append('../utils')
from data_loader import load_and_preprocess
from logger_utils import setup_logger
from model_architectures import build_model

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logger = setup_logger('training_logger', config['logging']['log_file'])
    logger.info("Starting MedVis Diagnostica pipeline...")

    images, labels = load_and_preprocess(
        config['data']['image_dir'],
        config['data']['labels_csv'],
        tuple(config['data']['target_size']),
        augment=True,
        config=config
    )
    images = images / 255.0

    if images.size == 0:
    logger.error("No images found! Exiting.")
    return
    logger.info(f"Model architecture: {config['training']['architecture']}")

    model = build_model(
        input_shape,
        config['training']['architecture'],
        transfer_learning=config['model']['transfer_learning'],
        freeze_base=config['model']['freeze_base']
    )

    logger.info("Commencing training...")
    model.fit(images, labels, epochs=config['training']['epochs'], batch_size=config['training']['batch_size'], validation_split=0.2)
    logger.info("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/config.yaml')
    args = parser.parse_args()
    main(args.config)
