from torch import rand
from yaml import full_load

from initialize_custom_model import initialize_custom_model

if __name__ == '__main__':
    load_path = './configuration.yaml'
    with open(load_path) as f:
        config = full_load(f)

    models = {}
    for key in config['network_spec']:
        models[key] = initialize_custom_model(config['network_spec'][key])
        print(f"[{key} model]")
        print(models[key])

    model_input = rand((4, 3, 32, 32))
    model_output = models['cnn'](model_input)
    print('cnn input:', model_input.shape)
    print('cnn output:', model_output.shape)

    model_input = rand((4, 512))
    model_output = models['linear'](model_input)
    print('linear input:', model_input.shape)
    print('linear output:', model_output.shape)