from pipelines.experiment_tracking.experiment_tracking_pipeline import pytorch_experiment_tracking_pipeline


import click

@click.command()
@click.option('--learning_rate','-lr', default = 0.001, type=float, help='Learning rate of the model')
@click.option('--batch_size','-bs', default=8, show_default = True, help='Batch size of dataset')
@click.option('--epochs','-ep', default=20, show_default = True, help='Batch size of dataset')
def main(learning_rate, batch_size, epochs):

    # Initialize a pipeline run

    pipeline = pytorch_experiment_tracking_pipeline.with_options(run_name = "exp_track_{date}_{time}")

    pipeline(data_path='./data/data.txt', batch_size = batch_size, 
                                                learning_rate=learning_rate, epochs=epochs)


    # pytorch_experiment_tracking_pipeline(data_path='./data/data.txt', batch_size = batch_size, 
    #                                             learning_rate=learning_rate, epochs=epochs)



    return 0

if __name__ == "__main__":

    main()
