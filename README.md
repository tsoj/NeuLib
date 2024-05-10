# NeuLib

A Nim library for small, fully connected neural networks with support for sparse inputs.

### Create a model
```nim
var model = newNetwork(
    784,
    (40, relu),
    (40, relu),
    (10, sigmoid)
)
```

Creates a model with 784 input features, 10 sigmoid activated output features, and two hidden layers with 40 neurons, which are both ReLU activated.

The model will be randomly initialized using Kaiming initialization.

NeuLib provides several common activation functions: `sigmoid`, `relu`, `leakyRelu`, `identity`, and `tanh`.

### Execute a model

```nim
# Run the model with a dense input vector
var input: seq[Float]
...
let output = model.forward(input)

# Or pass a sparse input vector to the model
var sparseInput: seq[SparseElement]
...
let output = model.forward(sparseInput)
```

### Train a model

```nim
# First, create a new variable that can
# hold the necessary training information
var backpropInfo: BackpropInfo

for epoch in epochs:
  for (input, target) in data:
    # Execute the model, but also pass
    # backpropInfo to collect information
    # necessary for gradient calculation
    let output = model.forward(input, backpropInfo)
    let lossGradient = mseGradient(target, output)
    # Run a backwards pass to calculate the
    # gradient and add it to the model using a learning rate
    model.backward(lossGradient, backpropInfo, lr = 0.01)
```

### Store and load models

```nim
# Using JSON representation
writeFile("mnist_model.json", model.toJsonString)
var newModel = readFile("mnist_model.json").toNetwork

# Using binary representation (needs less disk space)
model.saveToFile "mnist_model.bin"
var newModel = loadNetworkFromFile "mnist_model.bin"
```

### Example

See [`train.nim`](./train.nim) for a small demo using the MNIST data set.  
Compile it with `nim r train.nim`.

### Notes for compiling

Use `-d:danger` and `--cc:clang` flags for optimal performance.



