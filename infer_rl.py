from jarvis.utils.trainer import Trainer, load_yaml_config


def main() -> None:

    config_dir: str = "config/training_config.yaml"
    # Load the YAML file
    config = load_yaml_config(config_dir)

    # Access each configuration component
    model_config = config.get('model_config', {})
    env_config = config.get('env_config', {})
    training_config = config.get('training_config', {})
    trainer = Trainer(model_config=model_config,
                      env_config=env_config,
                      training_config=training_config)
    trainer.generate_dataset(num_evals=150)


if __name__ == '__main__':
    main()
