import collections
import copy
import logging
import os
import random

from typing import Union, Callable, Dict, List

import evaluate
import numpy as np
import scipy.stats
import torch
import tqdm


logger = logging.getLogger(__name__)

# for sanity run
DEFAULT_INPUT_IMAGE = torch.randn(256, 256, 1)


def process_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def sem(L: List[float]) -> float:
    # standard error of mean
    result = scipy.stats.sem(np.array(L))
    if isinstance(result, np.ndarray):
        return result.mean().item()
    return result


def mean(L: Union[List[int], List[float]]) -> float:
    return sum(L) / len(L)


class BaseTrainer():
    def __init__(self, *args, **kwargs):
        self.process_logits_for_metrics = process_logits_for_metrics
        self.compute_metrics = self.compute_metrics_func
        self.metric_accuracy = evaluate.load("accuracy")
        self.args = args
        self.optimizer = torch.optim.Adam()
        
        self.general_kwargs = {
            "early_stopping": False
        }

    def sanity_encode_decode(self, input_image = None):
        """encodes and decodes an image as a sanity check"""
        if input_image is None:
            input_image = DEFAULT_INPUT_IMAGE
        self.model.eval() # where to set model
        print("=" * 16, "Begin trainer sanity check", "=" * 16)
        inputs = inputs.to(self.args.device)
        outputs = self.model(inputs)
        print("\tDecoded output shape ->", outputs.shape)
        # save output
        print("=" * 16, "End trainer sanity check", "=" * 16)

    def _log_preds_table(
            self, table_key: str, reconstructed_images: List[np.ndarray], original_images: List[np.ndarray]
    ):
        # we also need inferred latents here 
        if not self.args.use_wandb:
            return
        elif not (self.args.local_rank <= 0):
            # only show for main process
            return
        
        num_rows = 50
        idxs = random.choices(
            range(len(original_images)), k=min(len(original_images), num_rows)
        )

        import wandb

        data = []
        for idx in idxs:
            data.append([wandb.Image(original_images[idx]), wandb.Image(reconstructed_images[idx])])

        table = wandb.Table(columns=["Original", "Decoded"], data=data)
        wandb.log({table_key: table})

    
    def _get_decoded_images_and_latents(self, dataloader: torch.utils.data.DataLoader):
        assert not self.model.training

        all_preds = []
        all_labels = []
        for step, inputs in enumerate(
            tqdm.tqdm(dataloader, desc="generating from val", leave=False)
        ):
            inputs_cuda = {k: v.to(self.args.device) for k, v in inputs.items()}
            with torch.no_grad():
               reparam_z, hard_y = self.model.encoder.sample(inputs_cuda)
               outputs = self.model.decoder(reparam_z)
            
            all_preds.extend()
            all_labels.extend()

        return all_preds, all_labels

    def compute_metrics_func(self, preds, labels):

        assert (
            torch.tensor(preds).shape == torch.tensor(labels).shape
        ), f"preds.shape {preds.shape} / labels.shape {labels.shape}"

        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        accuracy_result = self.metric_accuracy.compute(
            predictions=preds, references=labels
        )

        return {**accuracy_result}


    def eval_generation_metrics(self, dataloader: torch.utils.data.DataLoader):
        # todo
        # log preds table self?
        # need call embedding model in utils
        preds_list, labels_list = self._get_decoded_images_and_latents(
            dataloader=dataloader
        )
        self._log_preds_table(
            table_key="val_image_preds",
            reconstructed_images=
            original_images=
        )

        # do original and reconstructed im sim computations with clip embs later
        metrics = self.compute_metrics_func(preds_list, labels_list)
        return metrics

    def evaluation_loop():
        """run eval and return metrics"""
        print()

    def _load_from_checkpoint():
        """in case we wanna modify models post-hoc; example: add dropout"""
        print()

    def training_step(self, model, inputs, **kwargs):
        
        labels = inputs.pop("labels")
        recon_x, kl, latent_y = model(**inputs)

        loss = self.compute_elbo_loss(recon_x, kl, labels, latent_y)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_elbo_loss(self, recon_x, x, kl, labels=None, latent_y=None):
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")
        elbo_loss = recon_loss + kl

        if labels is not None and latent_y is not None:
            supervised_loss = torch.nn.functional.cross_entropy(latent_y, labels)
            total_loss = elbo_loss + supervised_loss
        else:
            total_loss = elbo_loss

        return total_loss
    

        

    







