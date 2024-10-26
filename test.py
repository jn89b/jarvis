import yaml
from jarvis.datasets.base_dataset import BaseDataset

if __name__ == "__main__":
    data_config = "config/data_config.yaml"
    # Load the YAML file
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    print(data_config)
    dataset = BaseDataset(config=data_config, is_validation=False)
    # dataset.load_data()
    # print(dataset.data_loaded)
