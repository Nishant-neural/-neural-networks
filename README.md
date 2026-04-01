# Mini Deep Learning Framework (NumPy)

A modular deep learning framework built from scratch using NumPy, implementing forward and backward propagation, optimizers, and training pipelines similar to PyTorch.

## Features

* Modular layers (Linear, ReLU, Sigmoid, Dropout)
* Backpropagation engine
* Optimizers: SGD, Adam
* Mini-batch training
* Gradient checking
* Train / Eval mode
* Model save & load

## Example

```python
model = Sequential([
    Linear(2, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid()
])

model.fit(X, Y, epochs=1000)
```

## Results

* Successfully learns XOR
* Final loss < 0.01

## Project Structure

* `nn/` → core framework
* `examples/` → demos
* `tests/` → gradient checking

## Motivation

This project was built to understand how deep learning frameworks like PyTorch work internally.
