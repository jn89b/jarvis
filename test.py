import yaml
from jarvis.datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
if __name__ == "__main__":
    data_config = "config/data_config.yaml"
    # Load the YAML file
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    batch_size: int = 5
    dataset = BaseDataset(config=data_config, is_validation=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=dataset.collate_fn)

    print("len of dataloader: ", len(dataloader))

    for batch in dataloader:
        print(batch['batch_size'])
