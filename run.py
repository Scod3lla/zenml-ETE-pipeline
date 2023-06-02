from pipelines.experiment_tracking.experiment_tracking_pipeline import pytorch_experiment_tracking_pipeline


from steps.dataloader.dataloader_step import dataloader, DataLoadingParameters
from steps.load_model.load_model_step import load_model
from steps.train_evaluate.train_test_step import train_test, TrainingParameters

import click

@click.command()
@click.option('--learning_rate','-lr', default = 0.001, type=float, help='Learning rate of the model')
@click.option('--batch_size','-bs', default=8, show_default = True, help='Batch size of dataset')
@click.option('--epochs','-ep', default=20, show_default = True, help='Batch size of dataset')
def main(learning_rate, batch_size, epochs):

    # Initialize a pipeline run
    run_1 = pytorch_experiment_tracking_pipeline(
        load_data = dataloader(params=DataLoadingParameters(data_path='./data/data.txt', batch_size = batch_size)),
        load_model = load_model(),
        train_test = train_test(params= TrainingParameters(learning_rate=learning_rate, epochs=epochs))
    )


    run_1.run()


    return 0

if __name__ == "__main__":

    main()
