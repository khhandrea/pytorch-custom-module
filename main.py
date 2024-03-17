from torch import rand
from yaml import full_load

from custom_module import CustomModule

if __name__ == '__main__':
    load_path = './configuration.yaml'
    with open(load_path) as f:
        config = full_load(f)

    models = {}
    for key in config['network_spec']:
        models[key] = CustomModule(config['network_spec'][key])
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

    model_input = rand((128, 260))
    h_0 = rand((1, 256))
    output, h_n = models['gru'](model_input, h_0)
    print('gru input:', model_input.shape)
    print('gru output:', output.shape, h_n.shape)