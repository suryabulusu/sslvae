import abc
import hashlib
import json
import logging
import os
import sys
from typing import Dict, Optional

import datasets
import torch

from data_helpers import load_mnist_for_ssl
from models.config import Config
from models.vae import SSLVAE
from run_args import DataArguments, ModelArguments, TrainingArguments
from trainers.vae_trainer import VAE
from utils.utils import dataset_map_multi_worker, get_num_proc

logger = logging.getLogger(__name__)

# allow wandb to start slowly
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["_WANDB_STARTUP_DEBUG"] = "true"

DATASET_CACHE_PATH = os.environ.get("MNIST_CACHE", os.path.expanduser("~/.cache/mnist"))

# Noisy compilation from torch.compile
# see detailed logs of compilation
try:
    torch._logging.set_logs(dynamo=logging.INFO)
except AttributeError:
    # torch version too low
    pass


def md5_hash_kwargs(**kwargs) -> str:
    s = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()


class Experiment(abc.ABC):
    # abstract base class
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        training_args.metric_for_best_model = f"{data_args.dataset_name}_loss"

        logger.info(
            "Save checkpoints according to metric_for_best_model %s:",
            training_args.metric_for_best_model,
        )

        # Save all args.
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        # TODO: set random seed

        print(f"Experiment output_dir = {training_args.output_dir}")
        # Set up output_dir and wandb.
        self._setup_logging()
        self._consider_init_wandb()

    @property
    def config(self) -> Config:
        return Config(
            **vars(self.data_args),
            **vars(self.model_args),
            **vars(self.training_args),
        )

    @property
    def kwargs_hash(self) -> str:
        all_args = {
            **vars(self.model_args),
            **vars(self.data_args),
            **vars(self.training_args),
        }
        all_args.pop("local_rank")
        # print("all_args:", all_args)
        return md5_hash_kwargs(**all_args)

    @property
    def _is_main_worker(self) -> bool:
        return (self.training_args.local_rank <= 0) and (
            int(os.environ.get("LOCAL_RANK", 0)) <= 0
        )

    @property
    @abc.abstractmethod
    def _wandb_project_name(self) -> str:
        # experiment specific
        raise NotImplementedError()

    @property
    def _wandb_exp_name(self) -> str:
        name_args = [
            self.training_args.exp_group_name,
            self.training_args.exp_name,
            self.model_args.model_name_or_path,
        ]
        name_args = [n for n in name_args if ((n is not None) and len(n))]
        return "__".join(name_args)

    def _setup_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def run(self):
        if self.training_args.do_eval:
            self.evaluate()
        else:
            self.train()

    def train(self) -> Dict:
        # *** Training ***
        training_args = self.training_args
        logger.info("*** Training ***")

        # Log on each process a small summary of training.
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
            + f"fp16 training: {training_args.fp16}, bf16 training: {training_args.bf16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Checkpointing logic
        checkpoint = self._get_checkpoint()
        logging.info("Experiment::train() loaded checkpoint %s", checkpoint)
        trainer = self.load_trainer()

        # Save model_args and data_args before training. Trainer will save training_args.
        if training_args.local_rank <= 0:
            torch.save(
                self.data_args, os.path.join(training_args.output_dir, "data_args.bin")
            )
            torch.save(
                self.model_args,
                os.path.join(training_args.output_dir, "model_args.bin"),
            )

        print(f"train() called - resume-from_checkpoint = {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return metrics

    def evaluate(self) -> Dict:
        # eval
        logger.info("*** Evaluate ***")
        trainer = self.load_trainer()
        num_eval_samples = len(trainer.eval_dataset)
        metrics = trainer.evaluate()
        max_eval_samples = (
            self.data_args.max_eval_samples
            if self.data_args.max_eval_samples is not None
            else num_eval_samples
        )
        metrics["eval_samples"] = min(max_eval_samples, num_eval_samples)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    def _get_last_checkpoint(output_dir: str) -> Optional[str]:
        # assumes fol name starts with checkpoint
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint")
        ]
        return max(checkpoints, key=os.path.getmtime) if checkpoints else None

    def _get_checkpoint(self) -> Optional[str]:
        training_args = self.training_args
        last_checkpoint = None
        if (
            os.path.isdir(training_args.output_dir)
            and not training_args.overwrite_output_dir
        ):
            last_checkpoint = self.get_last_checkpoint(training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if checkpoint:
            logger.info("Loading from checkpoint %s", checkpoint)
        else:
            logger.info("No checkpoint found, training from scratch")

        return checkpoint

    def _consider_init_wandb(self) -> None:
        if self.training_args.use_wandb and self._is_main_worker:
            import wandb

            wandb.init(
                project=self._wandb_project_name,
                name=self._wandb_exp_name,
                id=self.kwargs_hash,
                resume=True,
            )
            training_args = vars(self.training_args)
            wandb.config.update(
                {
                    **vars(self.model_args),
                    **vars(self.data_args),
                    **training_args,
                },
                allow_val_change=True,
            )
        else:
            # Disable W&B
            pass
            # os.environ["WANDB_MODE"] = "disabled"
            # os.environ["WANDB_DISABLED"] = "true"

    @abc.abstractmethod
    def load_trainer(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError()

    def _load_train_dataset_uncached(self) -> datasets.DatasetDict:
        # train and validation
        data_args = self.data_args
        mnist = load_mnist_for_ssl(data_args)

        if data_args.use_less_data and data_args.use_less_data > 0:
            for key in mnist:
                new_length = min(
                    len(mnist[key], data_args.use_less_data)
                )  # need to edit
                mnist[key] = mnist[key].select(range(new_length))

        # here comes dataset map multi worker
        for key in mnist:
            # key: labeled, unlabeled, validation
            mnist = dataset_map_multi_worker(
                dataset=mnist,
                map_fn=lambda x: x,
                num_proc=get_num_proc(),
                batched=True,
                desc="preparing multi worker batches",
            )

        # mnist["train"].set_format("pt")

        return mnist

    def load_train_and_val_dataset(self):
        dataset_kwargs = {"dataset_name": self.data_args.dataset_name}  # mnist

        dataset_path = os.path.join(
            DATASET_CACHE_PATH, (md5_hash_kwargs(**dataset_kwargs) + ".arrow")
        )

        dataset_path = os.environ.get("MNIST_CACHE", dataset_path)

        if os.path.exists(dataset_path):
            print("loading train dataset from path:", dataset_path)
            mnist = datasets.load_from_disk(dataset_path)
        else:
            mnist = self._load_train_dataset_uncached()
            mnist.save_to_disk(dataset_path, max_shard_size="1GB")


class SSLExperiment(Experiment):
    @property
    def trainer_cls(self):
        return VAE

    @property
    def _wandb_project_name(self) -> str:
        return "ssl-vae-with-concrete"

    def load_model(self):
        return SSLVAE(config=self.config)

    def load_trainer(self):
        model = self.load_model()
        mnist = self.load_train_and_val_dataset()
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training model with name `{self.model_args.model_name_or_path}` - Total size={n_params/2**20:.2f}M params"
        )

        return self.trainer_cls(
            model=model,
            args=self.training_args,
            dataset=mnist,  # what about eval dataset
        )


class SSLExperimentPoole(SSLExperiment):
    @property
    def trainer_cls(self):
        # return VAEPoole -- straight-through estimator (hard sampling fwd pass)
        raise NotImplementedError()

    @property
    def _wandb_project_name(self) -> str:
        return "ssl-vae-poole"


EXPERIMENT_CLS_MAP = {
    "ssl_experiment_concrete_kl": SSLExperiment,
    "ssl_experiment_softmax_probs": SSLExperimentPoole,
}


def experiment_from_args(model_args, data_args, training_args) -> Experiment:
    if training_args.experiment in EXPERIMENT_CLS_MAP:
        experiment_cls = EXPERIMENT_CLS_MAP[training_args.experiment]  # type: ignore
    else:
        raise ValueError(f"Unknown experiment {training_args.experiment}")
    return experiment_cls(model_args, data_args, training_args)  # type: ignore
