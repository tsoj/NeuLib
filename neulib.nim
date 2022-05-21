type
    Float = float32
    ActivationFunction = object
        f: proc(x: Float): Float {.noSideEffect.}
        df: proc(x: Float): Float {.noSideEffect.}
    Layer = object
        bias: seq[Float]
        weights: seq[Float]
        numInputs: int
        numOutputs: int
        activation: ActivationFunction
    Network* = object
        layers: seq[Layer]
    Nothing = enum nothing
    LayerBackpropInfo = object
        layer: Layer
        preActivation: seq[Float]
        postActivation: seq[Float]
        inputGradient: seq[Float]
    BackpropInfo = object
        layers: seq[LayerBackpropInfo]
        input: seq[Float]

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
    assert description.len >= 2, "Network needs at least one input layer and one output layer"

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
    layer: Layer,
    input: openArray[Float],
    layerBackpropInfo: Nothing or var LayerBackpropInfo
): seq[Float] =
    result = layer.bias
    
    for inNeuron in 0..<layer.numInputs:
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            result[outNeuron] += layer.weights[i] * input[inNeuron]

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.preActivation = result

    for value in result.mitems:
        value = layer.activation.f(value)

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.postActivation = result

func backward(
    layer: Layer,
    outGradient: openArray[Float],
    inPostActivation: openArray[Float],
    layerBackpropInfo: var LayerBackpropInfo
) =

    assert layerBackpropInfo.layer.numOutputs == layer.numOutputs
    assert layerBackpropInfo.layer.bias.len == layer.numOutputs
    assert layerBackpropInfo.layer.numInputs == layer.numInputs
    assert layerBackpropInfo.layer.weights.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert inPostActivation.len == layer.numInputs
    assert layerBackpropInfo.inputGradient.len == layer.numInputs
    assert(block:
        var b = true
        for inNeuron in 0..<layer.numInputs:
            if layerBackpropInfo.inputGradient[inNeuron] != 0:
                b = false
        b
    )

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.layer.bias[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]

    for inNeuron in 0..<layer.numInputs:
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            layerBackpropInfo.layer.weights[i] += inPostActivation[inNeuron] * layerBackpropInfo.layer.bias[outNeuron]
            layerBackpropInfo.inputGradient[inNeuron] += layer.weights[i] * layerBackpropInfo.layer.bias[outNeuron]

func forward(
    network: Network,
    input: openArray[Float],
    backpropInfo: Nothing or var BackpropInfo
): seq[Float] =

    assert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    assert input.len == network.layers[0].numInputs, "Input size and input size of first layer must be the same"
    assert result.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"
    when backpropInfo isnot Nothing:
        assert backpropInfo.layers.len == network.layers.len

    result = forward(
        network.layers[0],
        input,
        when backpropInfo is Nothing: nothing else: backpropInfo.layers[0]
    )

    when backpropInfo isnot Nothing:
        backpropInfo.input = input.toSeq

    for i in 1..<network.layers.len:
        result = forward(
            network.layers[i],
            result,
            when backpropInfo is Nothing: nothing else: backpropInfo.layers[i]
        )

func forward*(
    network: Network,
    input: openArray[Float],
    backpropInfo: var BackpropInfo
): seq[Float] =
    network.forward(input, backpropInfo)

func forward*(
    network: Network,
    input: openArray[Float]
): seq[Float] =
    network.forward(input, nothing)

func backward*(
    network: Network,
    lossGradient: openArray[Float],
    backpropInfo: var BackpropInfo
) =

    assert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    assert lossGradient.len == network.layers[^1].numOutputs, "Loss size and output size of last layer must be the same"
    assert backpropInfo.layers.len == network.layers.len

    for i in countdown(network.layers.len - 1, 0):

        backward(
            network.layers[i],
            if i == network.layers.len - 1: lossGradient else: backpropInfo.layers[i + 1].inputGradient,
            if i == 0: backpropInfo.input else: backpropInfo.layers[i - 1].postActivation,
            backpropInfo.layers[i]
        )
    



#var model = newNetwork((1000, relu), (500, relu), (600, relu), (1, sigmoid))