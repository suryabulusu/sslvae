from run_args import DataArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser # really good at parsing
# although transfromers has nothing to do with sslvae

from experiments import experiment_from_args

def main():
    parser = HfArgumentParser(DataArguments, ModelArguments, TrainingArguments)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()

if __name__ == "__main__":
    main()