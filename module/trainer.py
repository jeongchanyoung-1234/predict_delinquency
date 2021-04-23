from copy import deepcopy

import numpy as np

import torch

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

SILNET = 0
BATCHWISE = 1
EPOCHWISE = 2


class KerasLikeEngine(Engine) :
    def __init__(self, func, model, optimizer, loss, config) :
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.config = config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, minibatch) :
        engine.model.train()

        x = minibatch
        x = x.to(engine.device)
        x = x.float()


        x_hat = engine.model(x)
        loss_i = engine.loss(x_hat.squeeze(), x)

        ### ---- train only---- ###
        engine.optimizer.zero_grad()
        loss_i.backward()
        engine.optimizer.step()
        ### ---- train only---- ###

        return {
            'loss' : float(loss_i),
        }


    @staticmethod
    def attach(train_engine, verbose=EPOCHWISE) :
        def attach_running_average(engine, metric_name) :
            RunningAverage(output_transform=lambda x : x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss']

        for metric_name in training_metric_names :
            attach_running_average(train_engine, metric_name)

        if verbose == BATCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, ['loss'])

        if verbose == EPOCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, ['loss'])

            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_logs(engine) :
                print('Epoch {} Train - Loss: {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['loss'],
                ))


    @staticmethod
    def check_best(engine) :
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss :
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, config, **kwargs) :
        torch.save({
            'model' : engine.best_model,
            'config' : config,
            **kwargs
        }, config.model_fn)


class Trainer :
    def __init__(self, config) :
        self.config = config

    def train(self, model, optimizer, loss, train_loader) :
        # Make Engine object
        train_engine = KerasLikeEngine(KerasLikeEngine.train, model, optimizer, loss, self.config)

        # Attach Metrics
        KerasLikeEngine.attach(train_engine, verbose=self.config.verbose)

        # Event Handling
        train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                     KerasLikeEngine.check_best)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                     KerasLikeEngine.save_model, train_engine, self.config)


        # running
        train_engine.run(train_loader, max_epochs=self.config.n_epochs)

        model.load_state_dict(train_engine.best_model)

        return model