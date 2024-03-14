# pytorch-initialize-module-from-yaml

- Python function which returns pytorch.nn.module according to configuration yaml file

## Rules
- def initialize_custom_model(spec: dict) -> pytorch.nn.Module
- 'spec' must have form of below. Be careful of RNN input/output!

```
initialization: bool
layers: [layer_element]
```

- 'layer_element' must have some forms of:
```
layer: 'linear'
spec: [INPUT_SIZE, OUTPUT_SIZE]
activation: activation
```

```
layer: 'conv2d'
spec: [IN_CHANNEL, OUT_CHANNEL, KERNEL_SIZE, STRIDE, PADDING]
activation: activation
```

```
layer: 'rnn'
spec: [INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS]
activation: activation
```

```
layer: 'lstm'
spec: [INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS]
activation: activation
```

```
layer: 'flatten'
```

- 'activation' must have form of one of:
```
- relu
- elu
- softmax
- none
```



## Dependency
- Numpy
- PyTorch
- PyYaml

## Run
```python
python3 main.py
```