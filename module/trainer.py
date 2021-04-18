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

        x, y = minibatch
        x, y = x.to(engine.device), y.to(engine.device)


        y_hat = engine.model(x)
        loss_i = engine.loss(y_hat.squeeze(), y)

        ### ---- train only---- ###
        engine.optimizer.zero_grad()
        loss_i.backward()
        engine.optimizer.step()
        ### ---- train only---- ###

        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor) :
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else :
            accuracy = 0

        return {
            'loss' : float(loss_i),
            'accuracy' : float(accuracy),
        }

    @staticmethod
    def valid(engine, minibatch) :
        engine.model.eval()

        with torch.no_grad() :
            x, y = minibatch
            x, y = x.to(engine.device), y.to(engine.device)


            y_hat = engine.model(x)
            loss_i = engine.loss(y_hat.squeeze(), y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor) :
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else :
                accuracy = 0

            return {
                'loss' : float(loss_i),
                'accuracy' : float(accuracy),
            }

    @staticmethod
    def attach(train_engine, val_engine, verbose=EPOCHWISE) :
        def attach_running_average(engine, metric_name) :
            RunningAverage(output_transform=lambda x : x[metric_name]).attach(
                engine,
                metric_name,
            )

        training_metric_names = ['loss', 'accuracy']

        for metric_name in training_metric_names :
            attach_running_average(train_engine, metric_name)

        if verbose == BATCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, ['loss', 'accuracy'])

        if verbose == EPOCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, ['loss', 'accuracy'])

            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_logs(engine) :
                print('Epoch {} Train - Accuracy: {:.4f} Loss: {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['accuracy'],
                    engine.state.metrics['loss'],
                ))

        validation_metric_names = ['loss', 'accuracy']

        for metric_name in validation_metric_names :
            attach_running_average(val_engine, metric_name)

        if verbose == BATCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(val_engine, ['loss', 'accuracy'])

        if verbose == EPOCHWISE :
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(val_engine, ['loss', 'accuracy'])

            # @val_engine.on(Events.EPOCH_COMPLETED)
            # def print_logs(engine) :
            #     print('        Valid - Accuracy: {:.4f} Loss: {:.4f} Lowest_loss={:.4f}'.format(
            #         engine.state.metrics['accuracy'],
            #         engine.state.metrics['loss'],
            #         engine.best_loss,
            #     ))

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

    @staticmethod
    def print_logs(engine) :
        print('        Valid - Accuracy: {:.4f} Loss: {:.4f} Lowest_loss={:.4f}'.format(
            engine.state.metrics['accuracy'],
            engine.state.metrics['loss'],
            engine.best_loss,
        ))


class Trainer :
    def __init__(self, config) :
        self.config = config

    def train(self, model, optimizer, loss, train_loader, val_loader) :
        # Make Engine object
        train_engine = KerasLikeEngine(KerasLikeEngine.train, model, optimizer, loss, self.config)
        val_engine = KerasLikeEngine(KerasLikeEngine.valid, model, optimizer, loss, self.config)

        # Attach Metrics
        KerasLikeEngine.attach(train_engine, val_engine, verbose=self.config.verbose)

        # Event Handling
        def run_validation(engine, validation_engine, valid_loader) :
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                       run_validation, val_engine, val_loader)
        val_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                     KerasLikeEngine.check_best)
        val_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                     KerasLikeEngine.save_model, train_engine, self.config)
        val_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                     KerasLikeEngine.print_logs, val_engine)


        # running
        train_engine.run(train_loader, max_epochs=self.config.n_epochs)

        model.load_state_dict(val_engine.best_model)

        return model