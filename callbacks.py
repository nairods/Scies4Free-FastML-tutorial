'''
Created on 7 Apr 2017

@author: jkiesele
'''

import json

# loss per epoch
from time import time

from tensorflow.keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


class newline_callbacks_begin(Callback):
    def __init__(self, outputDir, model_tag='model'):
        self.outputDir = outputDir
        self.model_tag = model_tag
        self.loss = []
        self.val_loss = []
        self.full_logs = []

    def on_epoch_end(self, epoch, epoch_logs={}):
        import os

        lossfile = os.path.join(self.outputDir, f'{self.model_tag}_losses.log')
        print('\n***callbacks***\nsaving losses to ' + lossfile)

        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))

        with open(lossfile, 'w') as f:
            for i in range(len(self.loss)):
                f.write(str(self.loss[i]))
                f.write(" ")
                f.write(str(self.val_loss[i]))
                f.write("\n")

        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)

        infofile = os.path.join(self.outputDir, f'{self.model_tag}_full_info.log')
        with open(infofile, 'w') as out:
            out.write(json.dumps(self.full_logs))


class newline_callbacks_end(Callback):
    def on_epoch_end(self, epoch, epoch_logs={}):  # noqa: B006
        print('\n***callbacks end***\n')


class Losstimer(Callback):
    def __init__(self, every=5):
        self.points = []
        self.every = every

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs):
        if (batch % self.every) != 0:
            return
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.items():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)


class all_callbacks:
    def __init__(
        self,
        stop_patience=10,
        lr_factor=0.5,
        lr_patience=1,
        lr_epsilon=0.001,
        lr_cooldown=4,
        lr_minimum=1e-5,
        outputDir='',
        model_tag='model'
    ):
        self.nl_begin = newline_callbacks_begin(outputDir, model_tag)
        self.nl_end = newline_callbacks_end()

        self.stopping = EarlyStopping(
            monitor='val_loss',
            patience=stop_patience,
            verbose=1,
            mode='min'
        )

        self.reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            mode='min',
            verbose=1,
            min_delta=lr_epsilon,
            cooldown=lr_cooldown,
            min_lr=lr_minimum,
        )

        self.modelbestcheckweights = ModelCheckpoint(
            outputDir + f"/{model_tag}_best.weights.h5",
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )

        self.modelcheckweights = ModelCheckpoint(
            outputDir + f"/{model_tag}_last.weights.h5",
            verbose=1,
            save_weights_only=True
        )

        self.modelcheckperiodweights = ModelCheckpoint(
            outputDir + f"/{model_tag}_epoch" + "{epoch:02d}.weights.h5",
            verbose=1,
            save_freq='epoch',
            save_weights_only=True
        )

        self.tb = TensorBoard(log_dir=outputDir + f'/{model_tag}_logs')

        self.history = History()
        self.timer = Losstimer()

        self.callbacks = [
            self.nl_begin,
            self.modelbestcheckweights,
            self.modelcheckweights,
            self.modelcheckperiodweights,
            self.reduce_lr,
            self.stopping,
            self.nl_end,
            self.tb,
            self.history,
            self.timer,
        ]
