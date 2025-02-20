
import argparse
import os

import google.cloud.aiplatform as aiplatform

parser = argparse.ArgumentParser()
# Args for experiment
parser.add_argument('--experiment', dest='experiment',
                    required=True, type=str,
                    help='Name of experiment')
parser.add_argument('--run', dest='run',
                    required=True, type=str,
                    help='Name of run within the experiment')

# Hyperparameters for experiment
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')

parser.add_argument('--dataset-uri', dest='dataset_uri',
                    required=True, type=str,
                    help='Location of the dataset')

parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv("AIP_MODEL_DIR"), type=str,
                    help='Storage location for the model')
args = parser.parse_args()

def get_data(dataset_uri, execution):
    # get the training data

    dataset_artifact = aiplatform.Artifact.create(
        schema_title="system.Dataset", display_name="example_dataset", uri=dataset_uri
    )

    execution.assign_input_artifacts([dataset_artifact])

    return None

def get_model():
    # get or create the model architecture
    return None

def train_model(dataset, model, epochs):
    aiplatform.log_params({"epochs": epochs})
    # train the model
    return model

def save_model(model, model_dir, execution):
    # save the model

    model_artifact = aiplatform.Artifact.create(
        schema_title="system.Model", display_name="example_model", uri=model_dir
    )
    execution.assign_output_artifacts([model_artifact])

# Create a run within the experiment
aiplatform.init(experiment=args.experiment)
aiplatform.start_run(args.run)

with aiplatform.start_execution(
    schema_title="system.ContainerExecution", display_name="example_training"
) as execution:
    dataset = get_data(args.dataset_uri, execution)
    model = get_model()
    model = train_model(dataset, model, args.epochs)
    save_model(model, args.model_dir, execution)

    # Store the lineage link in the experiment
    aiplatform.log_metrics({"lineage": execution.get_output_artifacts()[0].lineage_console_uri})

aiplatform.end_run()
