from pipelines.experiment_tracking.experiment_tracking_pipeline import pytorch_experiment_tracking_pipeline


from steps.dataloader.dataloader_step import (dataloader, DataLoadingParameters)
from steps.load_model.load_model_step import load_model
from steps.train_evaluate.train_test_step import train_test




if __name__ == "__main__":

    # Initialize a pipeline run
    run_1 = pytorch_experiment_tracking_pipeline(
        load_data = dataloader(params=DataLoadingParameters(data_path='./data/data.txt')),
        load_model = load_model(),
        train_test = train_test()
    )


    run_1.run(unlisted=True)