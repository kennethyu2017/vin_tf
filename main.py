"""
implement a CNN network as mentioned in VIN paper.
Author: kenneth yu
"""
import tensorflow as tf
from train import Trainer
import time
TRAINING_CFG = tf.app.flags.FLAGS  # alias



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("@@@  start grid domain-vin_tf - {}  @@@ start time:{}".format('training' if TRAINING_CFG.is_training
                                                                                   else 'testing', time.ctime()))
    trainer = Trainer()
    if TRAINING_CFG.is_training:
      trainer.train()
    else:
      trainer.predict_and_show()
