import os
import sys

import random
import time
import torch
import numpy as np
from dcase_util.utils import logging
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler

import shared_globals
from helpers.utils import AverageMeter, DictAverageMeter, my_mixup, get_criterion, get_total_evaluation, \
    get_evaluation, create_optimizer, load_model, swa_moving_average, bn_update, count_parameters, Event, worker_init_fn
from attrdict import AttrDefault
from tensorboardX import SummaryWriter

from datasets import DatasetsManager
logger = shared_globals.logger


class Trainer:
    # Events
    eventAfterEpoch = Event()
    eventAfterTrainingDataset = Event()
    eventAfterTestingDataset = Event()

    def __init__(self, config, seed=42):
        global logger
        logger = shared_globals.logger
        config = AttrDefault(lambda: None, config)

        self.config = config
        self.datasets = {}
        self.data_loaders = {}
        self.use_swa = config.use_swa
        #self.run.info['epoch'] = 0
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed + 1)
        random.seed(seed + 2)

        self.min_lr = self.config.optim_config["min_lr"]
        if self.min_lr is None:
            self.min_lr = 0.0
        print(self.min_lr)
        # making outout dirs
        models_outputdir = os.path.join(config.out_dir, "models")
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)
        if not os.path.exists(models_outputdir):
            os.makedirs(models_outputdir)
        #self.run.info['out_path'] = config.out_dir


        # init_loggers
        self.init_loggers()


        self.dataset_manager= DatasetsManager(self.config['audiodataset'])


        # init Tensor board
        if self.config.tensorboard:
            tensorboard_write_path = config.tensorboard_write_path
            if not tensorboard_write_path:
                tensorboard_write_path = self.config.out_dir.replace("out", "runs", 1)
            shared_globals.console.info("tensorboard run path: " + tensorboard_write_path)
            shared_globals.console.info(
                "To monitor this experiment use:\n " + shared_globals.bcolors.FAIL +
                "tensorboard --logdir " + tensorboard_write_path + shared_globals.bcolors.ENDC)
            #self.run.info['tensorboard_path'] = tensorboard_write_path
            self.writer = SummaryWriter(
                tensorboard_write_path)

        # init multi gpu
        self.bare_model = load_model(config.model_config)
        print(self.bare_model)
        if self.use_swa:
            self.swa_model = load_model(config.model_config)
            if self.config.use_gpu:
                self.swa_model.cuda()
            self.swa_n = 0
            self.swa_c_epochs = config.swa_c_epochs
            self.swa_start = config.swa_start
        if self.config.use_gpu:
            self.bare_model.cuda()

        shared_globals.console.info("\n\nTrainable model parameters {}, non-trainable {} \n\n".format(
            count_parameters(self.bare_model), count_parameters(self.bare_model, False)))
        # DataParallel mode
        if not config.parallel_mode:
            self.model = self.bare_model
        elif config.parallel_mode == "distributed":
            torch.distributed.init_process_group(backend='nccl',
                                                 world_size=1, rank=0,
                                                 init_method='file://' + config.out_dir + "/shared_file")
            self.model = torch.nn.parallel.DistributedDataParallel(self.bare_model)
        else:
            self.model = torch.nn.DataParallel(self.bare_model)
        # self.model.cuda()

        # if load_model

        if config.get('load_model'):
            load_model_path = config.get('load_model')
            load_model_path = os.path.expanduser(load_model_path)
            shared_globals.console.info("Loading model located at: " + load_model_path)
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.use_swa:
                swa_state_dict = checkpoint.get('swa_state_dict', None)
                self.swa_n = checkpoint.get('swa_n', 0)
                if (swa_state_dict is not None) and not self.config.swa_model_load_same:
                    self.swa_model.load_state_dict(swa_state_dict)
                else:
                    shared_globals.console.warning("No swa_state_dict loaded! same loaded")
                    self.swa_model.load_state_dict(checkpoint['state_dict'])
                    self.swa_n = 0

        shared_globals.logger.info(str(self.model))
        shared_globals.current_learning_rate = config.optim_config['base_lr']
        self.optimizer, self.scheduler = create_optimizer(self.model.parameters(), config.optim_config)
        print("optimizer:", self.optimizer)
        loss_criterion_args = dict(config.loss_criterion_args)
        self.criterion = get_criterion(config.loss_criterion)(**loss_criterion_args)

        # init state
        inf_value = -float("inf")
        if self.config["optim_config"].get("model_selection", {}).get("select_min", False):
            inf_value = float("inf")
        self.state = {
            # 'config': self.config,
            'state_dict': None,
            'optimizer': None,
            'epoch': 0,
            'metrics': {},
            'best_metric_value': inf_value,
            'best_epoch': 0,
        }
        self.first_batch_done = False
        # init dataset loaders
        self.init_loaders()

        if config.get('load_model'):
            if not config.get("load_model_no_test_first"):
                testing_result = {}
                for name in self.config.datasets:
                    dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
                    if dataset_config.testing:
                        testing_result[name] = self.test(0, name, dataset_config)

                # updating the state with new results
                self.update_state(testing_result, 0)

    def init_loaders(self):
        # maybe lazy load for predicting only runs
        for name in self.config.datasets:
            dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
            if self.config['predict_only_mode'] and not dataset_config.predicting:
                continue
            # ds = self.run.get_command_function(dataset_config.dataset)()
            ds = self.dataset_manager.get_dataset(dataset_config)
        
            self.datasets[name] = ds
            shared_globals.logger.info("Initialized Dataset  `" + name + "` with {} Samples ".format(len(ds)))
            if dataset_config.batch_config.get("batch_sampler") == "stratified":
                shared_globals.logger.info("Initializing  StratifiedBatchSampler for " + name)
                batch_sampler = StratifiedBatchSampler(ds, dataset_config.batch_config.batch_size, self.config.epochs)
            elif dataset_config.batch_config.get("batch_sampler") == "sequential":
                shared_globals.logger.info("Initializing Sequential Sampler for " + name)
                sampler = SequentialSampler(ds)
                batch_sampler = BatchSampler(sampler, dataset_config.batch_config.batch_size, False)
            else:
                if dataset_config.testing or dataset_config.predicting:
                    shared_globals.logger.info("Initializing Sequential Sampler for " + name)
                    sampler = SequentialSampler(ds)
                else:
                    shared_globals.logger.info("Initializing RandomSampler for " + name)
                    sampler = RandomSampler(ds)
                batch_sampler = BatchSampler(sampler, dataset_config.batch_config.batch_size, True)
            loader = torch.utils.data.DataLoader(
                ds,
                # batch_size=batch_size,
                batch_sampler=batch_sampler,
                # shuffle=True,
                num_workers=dataset_config.num_of_workers,
                pin_memory=True,
                # drop_last=True,
                worker_init_fn=worker_init_fn,
                timeout=60
            )
            self.data_loaders[name] = loader

    def fit(self, epochs, start_epoch=0):

        try:
            for epoch in range(start_epoch, epochs):
                # Training
                for name in self.config.datasets:
                    dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
                    if dataset_config.training:
                        if dataset_config.frequency and ((epoch + 1) % dataset_config.frequency):
                            continue
                        self.train(epoch, name, dataset_config)
                    # notify the model that training done
                    epoch_done_op = getattr(self.bare_model, "epoch_done", None)
                    if callable(epoch_done_op):
                        epoch_done_op(epoch)
                if self.use_swa and (epoch + 1) >= self.use_swa and (
                        epoch + 1 - self.use_swa) % self.swa_c_epochs == 0:
                    swa_moving_average(self.swa_model, self.bare_model, 1.0 / (self.swa_n + 1))
                    self.swa_n += 1
                    if not self.config["swa_no_bn_update"]:
                        bn_update(self.data_loaders['training'], self.swa_model)
                    self.state['swa_state_dict'] = self.swa_model.state_dict()
                    self.state['swa_n'] = self.swa_n
                    #self.run.info['swa_n'] = self.swa_n
                    self.save_model(epoch)
                    # Testing
                    swa_testing_result = {}
                    for name in self.config.datasets:
                        dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
                        if dataset_config.testing:
                            swa_testing_result[name] = self.test(epoch, name, dataset_config, model=self.swa_model,
                                                                 extra_name="_swa")

                # Testing
                testing_result = {}
                for name in self.config.datasets:
                    dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
                    if dataset_config.testing:
                        testing_result[name] = self.test(epoch, name, dataset_config)

                # updating the state with new results
                self.update_state(testing_result, epoch)

                #self.run.info['epoch'] = epoch
                self.eventAfterEpoch(self, epoch)

                if shared_globals.current_learning_rate < self.min_lr:
                    shared_globals.console.info("learning rate reached minimum {} ({}), stopping in epoch {}".
                                                format(self.min_lr, shared_globals.current_learning_rate, epoch))
                    break

        except KeyboardInterrupt:
            pass
        shared_globals.console.info("last test:\n" + str(self.state['metrics']))

    def train(self, epoch, dataset_name, dataset_config, model=None):
        logger.info('Train ({})  epoch {}:'.format(dataset_name, epoch))
        if model is None:
            model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer

        optim_config = self.config.optim_config
        model_config = self.config.model_config

        if self.config.tensorboard:
            writer = self.writer

        # training mode
        model.train()

        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        metrics_meter = DictAverageMeter()
        start = time.time()
        train_loader = self.data_loaders[dataset_name]
        start_loading_time = time.time()
        total_loading_time = 0

        if optim_config['scheduler'] == 'multistep':
            scheduler.step(epoch + 1)
        elif optim_config['scheduler'] == 'mycos':
            scheduler.step(epoch + 1)
        elif optim_config['scheduler'] == 'swa':
            scheduler.step(epoch + 1)
        elif optim_config['scheduler'] == 'linear':
            scheduler.step(epoch)
        elif optim_config['scheduler'] == 'drop':
            scheduler.step(epoch)

        number_of_steps = len(train_loader)
        if self.config.maximum_steps_per_epoch and self.config.maximum_steps_per_epoch < number_of_steps:
            number_of_steps = self.config.maximum_steps_per_epoch

        for step, (data, _, targets) in enumerate(train_loader):
            shared_globals.global_step += 1
            if optim_config['scheduler'] == 'cosine':
                scheduler.step()
            if self.config.use_gpu:
                data = data.cuda()
                targets = targets.cuda()
            if self.config.use_mixup and epoch >= int(self.config.use_mixup) - 1:
                # don't forget to use mix up loss
                rn_indices, lam = my_mixup(data, targets, self.config.mixup_alpha, self.config.get("mixup_mode"))
                if self.config.use_gpu:
                    rn_indices = rn_indices.cuda()
                    lam = lam.cuda()
                data = data * lam.reshape(lam.size(0), 1, 1, 1) \
                       + data[rn_indices] * (1 - lam).reshape(lam.size(0), 1, 1, 1)

            # data is loaded
            total_loading_time += time.time() - start_loading_time
            # Model graph to tensor board
            if not self.first_batch_done:
                self.first_batch_done = True
                if self.config.tensorboard and not self.config.tensorboard_no_model_graph:
                    shared_globals.console.info("writing model graph to tensorboard!")
                    self.writer.add_graph(self.bare_model, data[0:1])
            optimizer.zero_grad()

            outputs = model(data)

            if self.config.use_mixup and epoch >= int(self.config.use_mixup) - 1:
                loss = self.criterion(outputs, targets, targets[rn_indices], lam, self.config.get("mixup_mode"))
            else:
                # print("targets", targets)
                if model_config.binary_classifier:
                    targets = targets.float()  # https://discuss.pytorch.org/t/data-type-mismatch-in-loss-function/34860
                    # print("targets.float()", targets)
                loss = self.criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            if model_config['multi_label']:
                preds = (outputs > model_config['prediction_threshold']).float()
            elif model_config.binary_classifier:
                if model_config.sigmoid_output:
                    preds = outputs > 0.5
                else:
                    preds = outputs > 0.
            elif model_config.regression:
                preds = outputs
            else:
                _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()

            # if data_config['use_mixup']:
            #     _, targets = targets.max(dim=1)

            if model_config.binary_classifier:
                targets_binary = targets > 0.5  # this is to account for smoothed labels
                # smoothed labels like in [Schlüter 2015] who used [0.02, 0.98] instead of [0, 1]
                correct_ = preds.eq(targets_binary).sum().item()
            elif model_config.regression:
                # in regression accuracy is L1 loss
                correct_ = torch.abs(preds - targets).sum().item()
            else:
                correct_ = preds.eq(targets).sum().item()

            if model_config['multi_label']:
                num = data.size(0) * model_config['n_classes']
            else:
                num = data.size(0)

            accuracy = correct_ / num
            eval_metrics = {}
            for ef in dataset_config._mapping.get("evaluations", []):
                ev_func = get_evaluation(ef["name"])
                if epoch % ef.get("frequency", 1) == 0:
                    eval_metrics = {**eval_metrics, **ev_func(outputs, targets, eval_args=ef.get("eval_args", {}))}
            metrics_meter.update(eval_metrics, num)
            loss_meter.update(loss_, num)
            accuracy_meter.update(accuracy, num)

            if self.config.tensorboard:
                writer.add_scalar(dataset_name + '/RunningLoss', loss_, shared_globals.global_step)
                writer.add_scalar(dataset_name + '/RunningAccuracy', accuracy,
                                  shared_globals.global_step)
                writer.add_scalars(dataset_name + "/RunningMetrics", eval_metrics,
                                   shared_globals.global_step)
            if step % (number_of_steps // 10) == 0:
                print('\x1b[2K ' + 'Epoch {} Step {}/{} '
                                   'Loss {:.4f} ({:.4f}) '
                                   'Accuracy {:.4f} ({:.4f}) '.format(
                    epoch,
                    step + 1,
                    number_of_steps,
                    loss_meter.val,
                    loss_meter.avg,
                    accuracy_meter.val,
                    accuracy_meter.avg), end="\r")
            if step % 100 == 0:
                logger.info('Epoch {} Step {}/{} '
                            'Loss {:.4f} ({:.4f}) '
                            'Accuracy {:.4f} ({:.4f})'.format(
                    epoch,
                    step,
                    number_of_steps,
                    loss_meter.val,
                    loss_meter.avg,
                    accuracy_meter.val,
                    accuracy_meter.avg,
                ))
            # to get the data loading time
            start_loading_time = time.time()
            if self.config.maximum_steps_per_epoch and step + 1 == self.config.maximum_steps_per_epoch:
                break

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f} (loading: {:.2f} )'.format(elapsed, total_loading_time))
        logger.info('avg metrics:  {}'.format(str(metrics_meter.avg)))
        print('\x1b[2K' + 'Train[{}]{}:Step {}/{} '
                          'Loss {:.4f} ({:.4f}) '
                          'Accuracy {:.4f} ({:.4f})'.format(
            epoch, dataset_name,
            step + 1,
            number_of_steps,
            loss_meter.val,
            loss_meter.avg,
            accuracy_meter.val,
            accuracy_meter.avg), end="\r")
        eval_metrics = {"loss": loss_meter.avg, "accuracy": accuracy}
        for ef in dataset_config._mapping.get("total_evaluations", []):
            ev_func = get_total_evaluation(ef["name"])
            eval_metrics = {**eval_metrics,
                            **ev_func(metrics_meter, model=model, data_loader=train_loader, config=self.config,
                                      current_dataset_config=dataset_config,
                                      eval_args=ef.get("eval_args", {}))}
        logger.info('total metrics:  {}'.format(str(eval_metrics)))

        # logging metrics resutls
        #self.run.info.setdefault("last_metrics", {})[dataset_name] = eval_metrics
        # for k, v in eval_metrics.items():
        #     self.log_scalar(dataset_name + "." + k, v, epoch)

        if self.config.tensorboard:
            writer.add_scalar(dataset_name + '/Loss', loss_meter.avg, epoch)
            writer.add_scalar(dataset_name + '/Accuracy', accuracy_meter.avg, epoch)
            writer.add_scalar(dataset_name + '/Time', elapsed, epoch)
            writer.add_scalars(dataset_name + "/AvgMetrics", metrics_meter.avg, epoch)
            writer.add_scalars(dataset_name + "/TotalMetrics", eval_metrics, epoch)
            if optim_config.get('scheduler') and optim_config['scheduler'] != 'none':
                lr = scheduler.get_lr()[0]
            else:
                lr = optim_config['base_lr']
            writer.add_scalar(dataset_name + '/LearningRate', lr, epoch)
            #self.run.log_scalar("LearningRate", lr, epoch)

    def test(self, epoch, dataset_name, dataset_config, model=None, extra_name=""):
        logger.info('Testing on ({}) epoch {}:'.format(dataset_name + extra_name, epoch))

        if model is None:
            model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer

        optim_config = self.config.optim_config
        model_config = self.config.model_config
        if self.config.tensorboard:
            writer = self.writer
        # training mode
        model.eval()

        loss_meter = AverageMeter()
        correct_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        metrics_meter = DictAverageMeter()
        start = time.time()
        test_loader = self.data_loaders[dataset_name]
        dataset_name = dataset_name + extra_name
        for step, (data, _, targets) in enumerate(test_loader):

            if self.config.tensorboard_test_images:
                if epoch == 0 and step == 0:
                    image = torchvision.utils.make_grid(
                        data, normalize=True, scale_each=True)
                    writer.add_image(dataset_name + '/Image', image, epoch)

            if self.config.use_gpu:
                data = data.cuda()
                targets = targets.cuda()

            with torch.no_grad():
                outputs = model(data)

            if model_config.binary_classifier:
                targets = targets.float()  # https://discuss.pytorch.org/t/data-type-mismatch-in-loss-function/34860

            loss = self.criterion(outputs, targets)

            # if data_config['use_mixup']:
            #     _, targets = targets.max(dim=1)

            if model_config['multi_label']:
                preds = (outputs > model_config['prediction_threshold']).float()
            elif model_config.binary_classifier:
                if model_config.sigmoid_output:
                    preds = outputs > 0.5
                else:
                    preds = outputs > 0.
            elif model_config.regression:
                preds = outputs
            else:
                _, preds = torch.max(outputs, dim=1)
            loss_ = loss.item()

            if model_config.binary_classifier:
                targets_binary = targets > 0.5  # accounting for smoothed labels
                correct_ = preds.eq(targets_binary).sum().item()
            elif model_config.regression:
                # in regression accuracy is L1 loss
                correct_ = torch.abs(preds - targets).sum().item()
            else:
                correct_ = preds.eq(targets).sum().item()

            if model_config['multi_label']:
                num = data.size(0) * model_config['n_classes']
            else:
                num = data.size(0)

            if model_config['multi_label']:
                total_num = len(test_loader.dataset) * model_config['n_classes']
            else:
                total_num = len(test_loader.dataset)

            eval_metrics = {}
            for ef in dataset_config._mapping.get("evaluations", []):
                ev_func = get_evaluation(ef["name"])
                if epoch % ef.get("frequency", 1) == 0:
                    eval_metrics = {**eval_metrics, **ev_func(outputs, targets, eval_args=ef.get("eval_args", {}))}
            metrics_meter.update(eval_metrics, num)
            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)
            accuracy = correct_meter.sum / total_num
            accuracy_meter.update(accuracy, num)
            if step % ((len(test_loader) + 10) // 10) == 0:
                print('\x1b[2K', 'Test[{}]{}: Step {}/{} '
                                 'Loss {:.4f} ({:.4f}) '
                                 'Accuracy {:.4f} ({:.4f})'.format(
                    epoch, dataset_name,
                    step + 1,
                    len(test_loader),
                    loss_meter.val,
                    loss_meter.avg,
                    accuracy_meter.val,
                    accuracy_meter.avg), end="\r")
        print('\x1b[2K', 'Test[{}]{}:Step {}/{} '
                         'Loss {:.4f} ({:.4f}) '
                         'Accuracy {:.4f} ({:.4f})'.format(
            epoch, dataset_name,
            step + 1,
            len(test_loader),
            loss_meter.val,
            loss_meter.avg,
            accuracy_meter.val,
            accuracy_meter.avg), end="\r")

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))
        logger.info('avg metrics:  {}'.format(str(metrics_meter.avg)))

        eval_metrics = {"loss": loss_meter.avg, "accuracy": accuracy}
        for ef in dataset_config._mapping.get("total_evaluations", []):
            ev_func = get_total_evaluation(ef["name"])
            eval_metrics = {**eval_metrics,
                            **ev_func(metrics_meter, model=model, data_loader=test_loader, config=self.config,
                                      current_dataset_config=dataset_config,
                                      eval_args=ef.get("eval_args", {}))}
        logger.info('total metrics:  {}'.format(str(eval_metrics)))
        #self.run.info.setdefault("last_metrics", {})[dataset_name] = eval_metrics
        # for k, v in eval_metrics.items():
        #     self.run.log_scalar(dataset_name + "." + k, v, epoch)
        
        if self.config.tensorboard:
            writer.add_scalar(dataset_name + '/Loss', loss_meter.avg, epoch)
            writer.add_scalar(dataset_name + '/Accuracy', accuracy, epoch)
            writer.add_scalar(dataset_name + '/Time', elapsed, epoch)
            writer.add_scalars(dataset_name + "/AvgMetrics", metrics_meter.avg, epoch)
            writer.add_scalars(dataset_name + "/TotalMetrics", eval_metrics, epoch)
        return eval_metrics

    def init_loggers(self):
        shared_globals.logger = logging.getLogger('')
        while len(shared_globals.logger.handlers):
            shared_globals.logger.handlers.pop()
        shared_globals.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.config.out_dir + "/info.log")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(fmt='%(asctime)s %(name)-5s %(levelname)-.1s %(message)s', datefmt='%m-%d %H:%M'))

        shared_globals.logger.addHandler(fh)

        # prevent multioutput when creating multiple trainer instances!
        if shared_globals.console is None:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            # set a format which is simpler for console use
            formatter = logging.Formatter('%(levelname)-.1s: %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler to the root logger
            logging.getLogger('.console').addHandler(console)

            shared_globals.console = logging.getLogger('.console')

        shared_globals.console.info("for detailed run info use \n " + shared_globals.bcolors.FAIL +
                                    "tail -f " + self.config.out_dir + "/info.log" + shared_globals.bcolors.ENDC)
        global logger
        logger = shared_globals.logger

    def update_state(self, testing_result, epoch):
        state = self.state
        state['epoch'] = epoch
        state['metrics'] = testing_result
        state['state_dict'] = self.bare_model.state_dict()
        model_path = os.path.join(self.config.out_dir, "models", 'last_model_{}.pth'.format(epoch))
        if epoch > 250 and epoch % 5 == 0:
            print("saving at ", model_path)
            torch.save(state, model_path)

        selection_config = self.config["optim_config"].get("model_selection", {
            "metric": "accuracy",
            "validation_set": "val",
            "patience": 30
        })

        # update best accuracy
        is_it_the_newbest_model = testing_result[selection_config['validation_set']][selection_config['metric']] > \
                                  state[
                                      'best_metric_value']
        if selection_config.get("select_min", False):
            is_it_the_newbest_model = testing_result[selection_config['validation_set']][selection_config['metric']] < \
                                      state[
                                          'best_metric_value']
        if is_it_the_newbest_model:
            state['state_dict'] = self.bare_model.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            state['best_metric_value'] = testing_result[selection_config['validation_set']][selection_config['metric']]
            state['best_epoch'] = epoch
            shared_globals.console.info("Epoch {}, found a new best model on set '{}', with {} {}".format(
                epoch,
                selection_config['validation_set'], state['best_metric_value'], selection_config['metric']))
            state['best_metrics'] = testing_result
            state['patience_rest_epoch'] = epoch
            #self.run.info['best_metrics'] = testing_result
            #self.run.info['best_epoch'] = epoch
            model_path = os.path.join(self.config.out_dir, "models", 'model_{}.pth'.format(epoch))
            best_model_path = os.path.join(self.config.out_dir, "models", 'model_best_state.pth')
            torch.save(state, model_path)
            torch.save(state, best_model_path)
            #self.run.info['best_model_path'] = best_model_path
            #self.run.info['best_metric_value'] = state['best_metric_value']
            #self.run.info['best_metric_name'] = selection_config['validation_set'] + "." + selection_config['metric']
        else:
            # logger.info(
            #     "Model didn't improve {} for {} on validation set '{}', patience {} of {} (Best so far {} at epoch {} )".format(
            #         selection_config['metric'], global_run_unique_identifier,
            #         selection_config['validation_set'], str(global_patience_counter),
            #         str(selection_config['patience']), str(state['best_metric_value']), str(state['best_epoch'])))
            patience = selection_config['patience'] - epoch + state['patience_rest_epoch']
            if patience <= 0:
                lr_min_limit = self.config["optim_config"].get("model_selection", {}).get(
                    "lr_min_limit", None)
                if (lr_min_limit is None) or shared_globals.current_learning_rate > lr_min_limit:
                    shared_globals.current_learning_rate *= self.config["optim_config"].get("model_selection",
                                                                                            {}).get(
                        "lr_decay_factor", 1.)
                    if selection_config.get("load_optimizer_state"):
                        raise NotImplementedError()
                    else:
                        if self.use_swa:
                            shared_globals.console.warning("SWA doesn't support LR decay via patience")
                        optim_config = self.config['optim_config']
                        optim_config['base_lr'] = shared_globals.current_learning_rate
                        self.optimizer, self.scheduler = create_optimizer(self.model.parameters(),
                                                                          self.config.optim_config)
                else:
                    self.config["optim_config"]['model_selection']['no_best_model_reload'] = True
                best_model_path = os.path.join(self.config.out_dir, "models", 'model_best_state.pth')
                best_epoch_to_reload = "no_reload"
                if not self.config["optim_config"].get("model_selection", {}).get(
                        "no_best_model_reload", False):
                    checkpoint = torch.load(best_model_path)
                    self.bare_model.load_state_dict(checkpoint['state_dict'])
                    best_epoch_to_reload = state['best_epoch']
                state['patience_rest_epoch'] = epoch
                shared_globals.console.info("Patience out({}), Loaded from epoch {}, lr= {} ".format(
                    epoch,
                    best_epoch_to_reload, shared_globals.current_learning_rate))

    def load_best_model(self):
        shared_globals.console.info("Loading best model...")
        best_model_path = os.path.join(self.config.out_dir, "models", 'model_best_state.pth')
        checkpoint = torch.load(best_model_path)
        self.bare_model.load_state_dict(checkpoint['state_dict'])

    def save_model(self, epoch):
        model_path = os.path.join(self.config.out_dir, "models", 'swa_model_{}.pth'.format(epoch))
        torch.save(self.state, model_path)

    def save_loadable_model(self, config):
        # TODO: create directory if it does not exist
        import pickle
        model = self.model
        experiment_path, model_name = config['experiment_path'], config['model_name']
        model_path = os.path.join(experiment_path, model_name + '_state_dict.pth')
        config_path = os.path.join(experiment_path, model_name + '_config.pkl')
        torch.save(model.state_dict(), model_path)
        pickle.dump(config, open(config_path, 'wb'))

    def evaluate(self):
        model = self.model

        # TODO: compute predictions in this function (similar to train, test...)
        # this allows use "evaluations" in addition to "total_evaluations"
        # keep track inside a metrics_meter (so tp, fp, ... does not need to be computed in the eval function)

        for dataset_name in self.config.datasets:
            dataset_config = AttrDefault(lambda: None, self.config.datasets[dataset_name])
            if dataset_config.evaluating:
                print("evaluate on ", dataset_name)
                # TODO: do not allow "evaluations" because this is not called after every batch
                data_loader = self.data_loaders[dataset_name]
                eval_metrics = {}
                for ef in dataset_config._mapping.get("total_evaluations", []):
                    ev_func = get_total_evaluation(ef["name"])
                    eval_metrics = {**eval_metrics,
                                    **ev_func(None, model=model, data_loader=data_loader, config=self.config,
                                              current_dataset_config=dataset_config,
                                              eval_args=ef.get("eval_args", {}))}

                    # logger.info('total metrics:  {}'.format(str(eval_metrics)))
                shared_globals.console.info("evaluation " + dataset_name + ":\n" + str(eval_metrics))
                # if self.config.tensorboard:
                # writer = self.writer
                # writer.add_scalar(dataset_name + '/RunningLoss', loss_, shared_globals.global_step)
                # writer.add_scalar(dataset_name + '/RunningAccuracy', accuracy,
                #                  shared_globals.global_step)
                # writer.add_scalars(dataset_name + "/RunningMetrics", eval_metrics,
                #                   shared_globals.global_step)

    def predict(self, name_extra=""):
        import helpers.output_writers as ow

        model = self.model
        for name in self.config.datasets:
            dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
            if dataset_config.predicting:
                sid, out = self.do_predict(name, dataset_config, model)
                for owriter_name in dataset_config.writers:
                    owcnfg = dataset_config.writers[owriter_name]
                    ow.__dict__[owcnfg['name']](sid, out, self, name + name_extra, owriter_name, **owcnfg['args'])

        if self.use_swa:
            model = self.swa_model
            for name in self.config.datasets:
                dataset_config = AttrDefault(lambda: None, self.config.datasets[name])
                if dataset_config.predicting:
                    sid, out = self.do_predict(name, dataset_config, model)
                    for owriter_name in dataset_config.writers:
                        owcnfg = dataset_config.writers[owriter_name]
                        ow.__dict__[owcnfg['name']](sid, out, self, name, owriter_name + "_swa", **owcnfg['args'])

    def do_predict(self, dataset_name, dataset_config, model=None):
        logger.info('Predicting on ({}) :'.format(dataset_name))

        if model is None:
            model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer

        optim_config = self.config.optim_config
        model_config = self.config.model_config
        if self.config.tensorboard:
            writer = self.writer
        # training mode
        model.eval()

        loss_meter = AverageMeter()
        correct_meter = AverageMeter()
        metrics_meter = DictAverageMeter()
        start = time.time()
        test_loader = self.data_loaders[dataset_name]
        acc_sids = []
        acc_out = []
        for step, (data, sids, _) in enumerate(test_loader):

            if self.config.tensorboard_test_images:
                image = torchvision.utils.make_grid(
                    data, normalize=True, scale_each=True)
                writer.add_image(dataset_name + '/Image', image, 0)

            if self.config.use_gpu:
                data = data.cuda()

            with torch.no_grad():
                outputs = model(data).cpu()
            acc_sids += sids
            acc_out.append(outputs)
            if step % (len(test_loader) // 10) == 0:
                print('\x1b[2K', 'Predicting  Step {}/{} '.format(

                    step + 1,
                    len(test_loader),
                ), end="\r")

        elapsed = time.time() - start
        logger.info('Elapsed {:.2f}'.format(elapsed))

        return acc_sids, torch.cat(acc_out, 0)

    def ERF_generate(self, dataset_name="testing", dataset_config="", model=None, extra_name=""):
        logger.info('ERF_generate on ({}) :'.format(dataset_name + extra_name))

        if model is None:
            config = dict(self.config.model_config)
            config['stop_before_global_avg_pooling'] = True
            model = load_model(config, self.experiment)
            model.cuda()
            best_model_path = os.path.join(self.config.out_dir, "models", 'model_best_state.pth')
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
        # testing mode
        model.eval()
        loader = self.data_loaders[dataset_name]
        counter = 0
        accum = None
        for step, (data, _, targets) in enumerate(loader):
            data = data.cuda()
            data.requires_grad = True
            outputs = model(data)
            grads = torch.zeros_like(outputs)
            grads[:, :, grads.size(2) // 2, grads.size(3) // 2] = 1
            outputs.backward(grads)
            me = np.abs(data.grad.cpu().numpy()).mean(axis=0).mean(axis=0)
            if accum is None:
                accum = me
            else:
                accum += me
            counter += 1
        torch.save({"arr": accum, "counter": counter}, os.path.join(self.config.out_dir, 'ERF_dict.pth'))
        ERF_plot(accum, savefile=os.path.join(self.config.out_dir, 'erf.png'))
        self.experiment.add_artifact(os.path.join(self.config.out_dir, 'erf.png'), "erf.png", {"dataset": dataset_name})
        return True

# def do_train_epoch(epoch, model, optimizer, scheduler, train_criterion,
#                    train_loaders, config, writer, state):
#     for train_loader in train_loaders:
#         if not train_loader["config"].get("no_default_train", False):
#             train(epoch, model, optimizer, scheduler, train_criterion,
#                   train_loader["loader"], config, writer, train_loader["config"])
#     return state
