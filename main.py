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
        # print('model')
        # print(models[key])

    input1 = rand((4, 512))
    output1 = models['linear'](input1)
    print('cnn:', output1.shape)