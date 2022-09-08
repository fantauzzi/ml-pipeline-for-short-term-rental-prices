#!/usr/bin/env python
"""
The MLFlow project uses Weights and Biases
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger()


def go(args):
    output_file_name = 'clean_sample.csv'
    run = wandb.init(job_type='basic_cleaning')
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact

    artifact = run.use_artifact(args.input_artifact)
    file_path = artifact.file()
    min_price = args.min_price
    max_price = args.max_price
    df = pd.read_csv(file_path)
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    df.to_csv(output_file_name, index=False)
    logging.info(f'Saved cleaned data into {output_file_name}')

    artifact = wandb.Artifact(args.output_artifact,
                              type=args.output_type,
                              description=args.output_description)
    artifact.add_file((output_file_name))
    run.log_artifact(artifact)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Essential data cleaning')

    parser.add_argument(
        '--input_artifact',
        type=str,
        help='The raw data to be cleaned',
        required=True
    )

    parser.add_argument(
        '--output_artifact',
        type=str,
        help='The cleaned data (the output artifact).',
        required=True
    )

    parser.add_argument(
        '--output_type',
        type=str,
        help='The type for the output artifact.',
        required=True
    )

    parser.add_argument(
        '--output_description',
        type=str,
        help='A description for the output artifact.',
        required=True
    )

    parser.add_argument(
        '--min_price',
        type=float,
        help='The minimum price to consider.',
        required=True
    )

    parser.add_argument(
        '--max_price',
        type=float,
        help='The maximum price to consider.',
        required=True
    )

    args = parser.parse_args()

    go(args)
