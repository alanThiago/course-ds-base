import yaml
from typing import Text

import click
import pandas as pd

@click.command(help="generate new features from raw data")
@click.option('--config-path', type=click.Path(exists=True), help="Path to the config file.", required=True)
def featurize_data(config_path: Text):
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    dataset = pd.read_csv(config['data']['iris_dataset'])

    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    
    dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
    #     'sepal_length_in_square', 'sepal_width_in_square', 'petal_length_in_square', 'petal_width_in_square',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    dataset.to_csv(config['data']['iris_featurized'], index=False)


if __name__ == '__main__':
    featurize_data()
