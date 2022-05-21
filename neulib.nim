type
    Float = float32
    ActivationFunction = object
        f: proc(x: Float): Float
        df: proc(x: Float): Float
    Layer = object
        bias: seq[Float]
        weights: seq[Float]
        numInputs: int
        numOutputs: int
        activation: ActivationFunction
    Network* = object
        layers: seq[Layer]
    Nothing = enum nothing
    BackpropInfo = object
        discard # TODO

func getBackpropInfo*(network: Network): BackpropInfo =
    discard # TODO

func newLayer*(numInputs, numOutputs: int, activation: ActivationFunction): Layer =
    assert numInputs > 0 and numOutputs > 0

    Layer(
        bias: newSeq[Float](numOutputs),
        weights: newSeq[Float](numInputs * numOutputs),
        numInputs: numInputs,
        numOutputs: numOutputs,
        activation: activation
    )

func newNetwork*(description: varargs[tuple[numNeurons: int, activation: ActivationFunction]]): Network =
    for i in 1..<description.len:
        result.layers.add(newLayer(
            numInputs = description[i-1].numNeurons,
            numOutputs = description[i].numNeurons,
            activation = description[i-1].activation
        ))

func weightIndex(inNeuron, outNeuron, numInputs, numOutputs: int): int =
    assert inNeuron in 0..<numInputs
    assert outNeuron in 0..<numOutputs

    outNeuron + numOutputs * inNeuron;


func forward(
    network: Network,
    input: openArray[Float],
    output: var openArray[Float],
    backpropInfo: Nothing or var BackpropInfo
) =
    assert network.layers.len >= 2, "Network needs at least one input layer and one output layer"
    assert input.len == network.layers[0].numInputs, "Input size and input size of first layer must be the same"
    assert output.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"

func forward*(
    network: Network,
    input: openArray[Float],
    output: var openArray[Float],
    backpropInfo: var BackpropInfo
) =
    forward(network, input, output, backpropInfo)

func forward*(
    network: Network,
    input: openArray[Float],
    output: var openArray[Float]
) =
    network.forward(input, output, nothing)

func forward*(
    network: Network,
    input: openArray[Float]
): seq[Float] =
    result = newSeq[Float](network.layers[^1].numOutputs)
    network.forward(input, result)






#var model = newNetwork((1000, relu), (500, relu), (600, relu), (1, sigmoid))