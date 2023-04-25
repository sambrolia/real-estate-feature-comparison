import yaml

with open("config/config.yaml", "r") as f:
    try:
        config = yaml.safe_load(f)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)
    else:
        print("Config file successfully read")
