import yaml
from typing import Text

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command(help="train a model")
@click.option('--config-path', type=click.Path(exists=True), help="Path to the config file.", required=True)
def train_model(config_path: Text):
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    train_dataset = pd.read_csv(config['data']['trainset_path'])

    y_train = train_dataset.loc[:, 'target'].values.astype('int32')
    X_train = train_dataset.drop('target', axis=1).values.astype('float32')

    params = config['train']['clf_params']
    logreg = LogisticRegression(**params, random_state=config["base"]["random_state"])
    logreg.fit(X_train, y_train)


if __name__ == '__main__':
    train_model()
