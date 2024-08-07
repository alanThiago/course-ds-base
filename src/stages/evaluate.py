import json
import yaml
from typing import Text

import click
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from report.visualize import plot_confusion_matrix


@click.command(help="report metrics and plots")
@click.option('--config-path', type=click.Path(exists=True), help="Path to the config file.", required=True)
def evaluate_model(config_path: Text):
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    test_dataset = pd.read_csv(config['data']['testset_path'])
    logreg = joblib.load(config['train']['model_path'])

    y_test = test_dataset.loc[:, 'target'].values.astype('int32')
    X_test = test_dataset.drop('target', axis=1).values.astype('float32')

    prediction = logreg.predict(X_test)
    cm = confusion_matrix(prediction, y_test)
    f1 = f1_score(y_true = y_test, y_pred = prediction, average='macro')

    cm_image = plot_confusion_matrix(cm, ['setosa', 'versicolor', 'virginica'], normalize=False)

    metrics = {
        'f1': f1
    }
    with open(config['evaluate']['metrics_file'], 'w') as metrics_file:
        json.dump(metrics, metrics_file)

    cm_image.savefig(config['evaluate']['confusion_matrix_image'])


if __name__ == '__main__':
    evaluate_model()
