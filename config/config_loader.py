#load the yaml file
import yaml
#here we define a function to load the config file returning the config as a dictionary
def load_config(config_path=r"D:\krish_naik_data_science_course\NLP\customer_support_system\config\config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#print(load_config())