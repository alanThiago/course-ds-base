import yaml
from typing import Text

import click
from sklearn.datasets import load_iris


@click.command(help="Load raw data from sklearn iris")
@click.option('--config-path', type=click.Path(exists=True), help="Path to the config file.", required=True)
def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = load_iris(as_frame=True)
    dataset = data.frame

    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]

    dataset.to_csv(config['data']['iris_dataset'], index=False)


if __name__ == '__main__':
    data_load()
