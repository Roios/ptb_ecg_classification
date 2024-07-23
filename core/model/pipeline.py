import logging

logger = logging.getLogger(__name__)
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import onnx
import onnxruntime
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score, hamming_loss, jaccard_score,
                             precision_score, recall_score)

import core.model.dataset as crds
import core.model.ml_plots as mlplots
from core.utils import EarlyStopping, balance_input, setup_logging


class Pipeline():

    def __init__(self, model: torch.nn.Module, labels_name: List, seed: int = 42):
        """ Pipeline for model training and testing.

        Args:
            model (torch.nn.Module): Model
            labels_name (List[str]): Name of the labels
            seed (int, optional): Random seed. Default to 42.
        """

        self.model = model
        # results folder
        date = datetime.today().strftime('%Y_%m_%d')
        hour = datetime.today().strftime('%H_%M_%S')
        self.dst_dir = Path(f"results/{date}/{hour}_{self.model.model_name}")
        if not self.dst_dir.exists():
            self.dst_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initiating model")

        # check if GPU is available
        self.use_cuda = torch.cuda.is_available()
        logger.info(f"GPU available: {self.use_cuda}")
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        if self.use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.model.to(self.device)

        # fix all random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False

        # model info
        logger.info(f"Model architecture: {self.model.model_name}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):.0f}")

        self.labels_name = labels_name
        self.num_classes = len(self.labels_name)

    def train(self,
              train_ds: crds.ECGDataset,
              val_ds: crds.ECGDataset,
              epochs: int,
              batch_size: int,
              lr: float,
              patience: int = 7,
              delta_stop: float = 0,
              val_every: int = 1,
              half_precision: bool = False) -> None:
        """ Train and evaluates a model

        Args:
            train_ds (ECGDataset): Training dataset
            val_ds (ECGDataset): Validation dataset
            epochs (int): Number of epochs
            batch_size (int): Batch size
            lr (float): Maximum learning rate
            patience (int, optional): Number of epochs before early stopping. Defaults to 7.
            delta_stop (float, optional): Minimum delta for early stopping. Defaults to 0.
            val_every (int, optional): Validation step every X epochs. Defaults to 1.
            half_precision(bool, optional): Train with half precision. Defaults to False.
        """

        logger.info("Training start")
        # setup the early stop
        early_stopping = EarlyStopping(patience=patience,
                                       delta=delta_stop,
                                       path=self.dst_dir.joinpath("checkpoint.pt"),
                                       logger=logger)

        # create the dataloaders
        class_weights = train_ds.class_weights
        train_loader = crds.make_dataloader(ds=train_ds, batch_size=batch_size)
        val_loader = crds.make_dataloader(ds=val_ds, batch_size=batch_size)

        # training setup
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                        max_lr=lr,
                                                        steps_per_epoch=len(train_loader),
                                                        epochs=epochs)

        if class_weights is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device))

        predictions = []
        lr_evolution = []
        history = {"epoch": [], "train_time": [], "train_loss": [], "val_time": [], "val_loss": []}
        early_break = False
        val_step_counter = 0
        if half_precision:
            scaler = torch.cuda.amp.GradScaler()
            logger.info("Training with half-precision.")

        # Train and val loop
        for epoch in range(epochs):
            # train step
            train_loss = 0
            train_start = time.time()
            self.model.train()
            for param_group in optimizer.param_groups:
                lr_evolution.append(param_group['lr'])
            for data, target in train_loader:
                optimizer.zero_grad()
                if half_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data.to(self.device))
                        loss = loss_fn(output, target.to(self.device))
                else:
                    output = self.model(data.to(self.device))
                    loss = loss_fn(output, target.to(self.device))

                train_loss += loss

                if half_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

            train_end = time.time()
            train_loss /= len(train_loader.dataset)

            # val step
            val_loss = 0
            val_start = time.time()
            self.model.eval()
            if val_step_counter % val_every == 0 or epoch == epochs - 1 or early_stopping.early_stop:
                val_step_counter = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = self.model(data.to(self.device))

                        if epoch == epochs - 1 or early_stopping.early_stop:
                            predictions.append(torch.nn.functional.sigmoid(output))

                    val_loss += loss_fn(output, target.to(self.device))
                val_loss /= len(val_loader.dataset)
            else:
                val_step_counter += 1
            val_end = time.time()

            history["epoch"].append(1 + epoch)
            history["train_time"].append(train_end - train_start)
            history["train_loss"].append(train_loss.detach().cpu().item())
            history["val_time"].append(val_end - val_start)
            history["val_loss"].append(val_loss.detach().cpu().item())

            logger.info(", ".join([f"{k}: {v[-1]:.6f}" for k, v in history.items()]))

            if early_break:
                break

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                early_break = early_stopping.early_stop

        if early_stopping.early_stop:
            early_stopping.load_checkpoint(self.model)

        mlplots.plot_loss_evolution(history["train_loss"], history["val_loss"], dst_dir=self.dst_dir, show=False)

        total_time_min = sum(history["train_time"]) / 60
        logger.info(f"Total training time: {total_time_min:.2f} minutes")

        mlplots.plot_lr_evolution(lrs=lr_evolution, dst_dir=self.dst_dir, show=False)

        # metrics
        predictions = torch.cat(predictions).detach().cpu().numpy()

        self.val_thresholds = mlplots.plot_pr_curves(ground_truth=val_ds.labels_bin,
                                                     predictions=predictions,
                                                     labels_name=self.labels_name,
                                                     data_type="val",
                                                     dst_dir=self.dst_dir,
                                                     show=False)
        logger.info(f"Thresholds: {self.val_thresholds}")
        self.val_thresholds = np.array([self.val_thresholds[l] for l in self.labels_name])
        predictions = (predictions > self.val_thresholds).astype(int)

        mlplots.plot_cm_multilabel(ground_truth=val_ds.labels_bin,
                                   predictions=predictions,
                                   labels_name=self.labels_name,
                                   data_type="val",
                                   dst_dir=self.dst_dir,
                                   show=False)

        self.report(val_ds.labels_bin, predictions, self.labels_name)

    def evaluate(self, x_eval: np.ndarray, y_eval: np.ndarray, batch_size: int = 512) -> None:
        """ Evaluates a model

        Args:
            x_val (np.ndarray): Evaluation input data
            y_val (np.ndarray): Evaluation label data
            learning_rate (float): Maximum learning rate
            batch_size (int): Batch size
        """

        # create the dataloader

        val_loader, _ = self.make_dataloader(x_data=x_eval,
                                             y_data=y_eval,
                                             batch_size=batch_size,
                                             use_sampler=False,
                                             is_val=True)

        predictions = []

        self.model.eval()
        val_start = time.time()
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data.to(self.device))
                predictions.append(torch.nn.functional.sigmoid(output))
        val_end = time.time()
        total_time_min = (val_end - val_start) / 60
        logger.info(f"total time: {total_time_min:.2f} minutes")

        # # metrics
        # predictions = torch.cat(predictions).detach().cpu().numpy()

        # self.test_thresholds = self.plot_pr_curves(y_eval, predictions, label_names, data_type="test")
        # self.test_thresholds = np.array([self.test_thresholds[l] for l in label_names])
        # predictions = (predictions > self.test_thresholds).astype(int)

        # self.plot_cm_multilabel(y_eval, predictions, label_names, data_type="test")

        # self.report(y_eval, predictions, label_names)

    def predict(self, x: np.ndarray, batch_size: int = 512, logits: bool = False) -> np.ndarray:
        """ Predicts the classes from a given input

        Args:
            x (np.ndarray): ECG signals (batch size, dimensions, timesteps)
            batch_size (int, optional): Batch size. Defaults to 512.
            logits (bool, optional): To output the logits.

        Returns:
            np.ndarray: Model's predictions
        """

        if x.shape[0] < batch_size:
            batch_size = x.shape[0]

        pred_loader, _ = self.make_dataloader(x_data=x, batch_size=batch_size, use_sampler=False, is_val=False)

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in pred_loader:
                output = self.model(data[0].to(self.device))
                if logits:
                    predictions.append(output)
                else:
                    predictions.append(torch.nn.functional.sigmoid(output))
        predictions = torch.cat(predictions).detach().cpu().numpy()

        return predictions

    def export_to_onnx(self, input_example: np.ndarray, thresholds: np.ndarray = None) -> None:
        """ Convert the model to ONNX.

        Args:
            input_example (np.ndarray): Input data to test the converted model
            thresholds (np.ndarray, optional): Thresholds to confirm a class. Defaults to 0.5

        """
        model_path = Path(self.dst_dir.joinpath("checkpoint.pt"))
        model_name = f"{model_path.stem}.onnx"
        model_name = self.dst_dir.joinpath(model_name)
        x = torch.from_numpy(input_example).float().to(self.device)

        torch.onnx.export(
            self.model,
            x,
            model_name,
            verbose=False,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

        onnx_model = onnx.load(str(model_name))
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(model_name,
                                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # example output with ONNX model
        onnx_inputs = {ort_session.get_inputs()[0].name: input_example.astype(np.float32)}
        onnx_outs = ort_session.run(None, onnx_inputs)[0]

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if thresholds is None:
            thresholds = np.ones(self.num_classes, dtype=float) * 0.5

        for i in range(len(onnx_outs)):
            onnx_outs[i] = np.where(sigmoid(onnx_outs[i]) > thresholds, 1, 0)
            onnx_outs[i] = np.where(sigmoid(onnx_outs[i]) > thresholds, 1, 0)

        torch_outs = self.predict(input_example, logits=False)
        torch_outs = np.where(torch_outs > thresholds, 1, 0)

        try:
            for torch_out, onnx_out in zip(torch_outs, onnx_outs):
                torch_out = torch_out.astype(int)
                onnx_out = onnx_out.astype(int)
                # absolute_error = np.linalg.norm(torch_out - onnx_out)
                # relative_error = absolute_error / np.linalg.norm(torch_out)
                np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-02)
            logger.info("Exported model has been tested with ONNXRuntime, and the results match!")
        except AssertionError:
            logger.info("!!! ONNX exported model is not compatible with the original !!! \n")

    def report(self, y_true, y_pred, label_names):

        # micro: averages metrics across all classes, emphasizing overall performance
        # macro: averages metrics independently for each class, giving equal weight to each class
        accuracy = accuracy_score(y_true, y_pred)

        # proportion of predicted positive cases that are actually positive across all classes
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        # proportion of actual positive cases that are correctly predicted as positive across all classes
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # measures the fraction of labels that are incorrectly predicted
        h_loss = hamming_loss(y_true, y_pred)

        # measures similarity between the predicted and true label sets
        jaccard = jaccard_score(y_true, y_pred, average='samples')

        class_report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)

        logger.info("Overall Metrics:")
        logger.info(f"Accuracy: {accuracy:.3f} (higher is better)")
        logger.info(f"Precision (Micro): {precision_micro:.3f} (higher is better)")
        logger.info(f"Precision (Macro): {precision_macro:.3f} (higher is better)")
        logger.info(f"Precision (Weighted): {precision_weighted:.3f} (higher is better)")
        logger.info(f"Recall (Micro): {recall_micro:.3f} (higher is better)")
        logger.info(f"Recall (Macro): {recall_macro:.3f} (higher is better)")
        logger.info(f"Recall (Weighted): {recall_weighted:.3f} (higher is better)")
        logger.info(f"F1-Score (Micro): {f1_micro:.3f} (higher is better)")
        logger.info(f"F1-Score (Macro): {f1_macro:.3f} (higher is better)")
        logger.info(f"F1-Score (Weighted): {f1_weighted:.3f} (higher is better)")
        logger.info(f"Hamming Loss: {h_loss:.3f} (lower is better)")
        logger.info(f"Jaccard Score: {jaccard:.3f} (higher is better)")

        logger.info(f"Classification Report:\n{class_report}")


if __name__ == "__main__":
    import core.dataloader as crloader
    from core.model.architectures import FullyConvolutionalNetwork

    logger = setup_logging("training_pipeline.log")

    epochs = 50
    batch_size = 128
    learning_rate = 0.001

    logger.info("Loading data...")
    data = crloader.load_data(
        data_path='C:/Users/pedro/Desktop/ptb_ecg_classification/data/physionet.org/files/ptb-xl/1.0.2',
        sampling_rate=100)
    logger.info("Data loaded.")

    for aug_type in [None, "under", "over"]:
        train_ds = crds.ECGDataset(data=data, apply_sampler=False, ds_type="train", shuffle=True, augmentation=aug_type)
        val_ds = crds.ECGDataset(data=data, apply_sampler=False, ds_type="val", shuffle=True)
        test_ds = crds.ECGDataset(data=data, apply_sampler=False, ds_type="test")

        model = FullyConvolutionalNetwork(num_classes=train_ds.num_classes,
                                          channels=train_ds.channels,
                                          filters=[128, 256, 128],
                                          kernel_sizes=[8, 5, 3],
                                          linear_layer_len=128)

        pipe = Pipeline(model=model, labels_name=train_ds.labels_class)

        pipe.train(train_ds=train_ds,
                   val_ds=val_ds,
                   epochs=epochs,
                   batch_size=batch_size,
                   lr=learning_rate,
                   patience=10,
                   delta_stop=0,
                   val_every=1,
                   half_precision=False)
