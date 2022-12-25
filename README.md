# NeuLib

A Nim library for small, fully connected neural networks with support for sparse inputs.

### Prerequisites

- You need to install [NimBLAS](https://github.com/SciNim/nimblas) (e.g. using Nimble: `nimble install nimblas`)
- Install a [BLAS](https://de.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) library such as OpenBLAS (I suggest not using a multihread BLAS implementation)
- When compiling your nim program, you may need to specify the BLAS library you want to use by adding `-d:blas=your_BLAS_library`. When, for example, you want to link `libopenblas.so.3` on Linux, you should pass to Nim the option `-d:blas=openblas`

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

NeuLib already provides several common activation functions: `sigmoid`, `softplus`, `silu`, `relu`, `elu`, `leakyRelu`, `identity`, `tanh`. Custom activation functions can be added using `newActivationFunction(f, df, name)`.

### Execute a model

```nim
# run the model with a dense input vector
var input: seq[Float]
...
let output = model.forward(input)

# or pass a sparse input vector to the model
var sparseInput: seq[SparseElement]
...
let output = model.forward(sparseInput)
```

### Train a model

```nim
# first, create a new variable that can
# hold the necessary training information
var backpropInfo = model.newBackpropInfo()

for each epoch:
    for batch in batches:
        # before each new gradient calculation,
        # reset backpropInfo
        backpropInfo.setZero

        for (input, target) in batch:
            # execute the model, but also pass
            # backpropInfo to collect information
            # necessary for gradient calculation
            let output = model.forward(input, backpropInfo)
            let lossGradient = mseGradient(target, output)
            # run a backwards pass to create the
            # gradient and add it to backpropInfo
            model.backward(lossGradient, backpropInfo)

        # apply the calculated gradient to the model
        model.addGradient(backpropInfo, lr = 0.2)
```

### Store and load models

```nim
writeFile("mnist_model.json", model.toJsonString)
var newModel = readFile("mnist_model.json").toNetwork
```

### Example

See [`train.nim`](./train.nim) for a small demo using the MNIST data set.  
Compile it with `nim r train.nim`.

### Notes for compiling

Use `-d:danger` and `-cc:clang` flags for optimal performance.



