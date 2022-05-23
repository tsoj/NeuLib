import
    math,
    sequtils

type
    Float = float32
    ActivationFunction = object
        f: proc(x: Float): Float {.noSideEffect.}
        df: proc(x: Float): Float {.noSideEffect.}
        name: string
    Layer = object
        bias: seq[Float]
        weights: seq[Float]
        numInputs: int
        numOutputs: int
        activation: ActivationFunction
    Network* = object
        layers: seq[Layer]
    Nothing = enum a
    LayerBackpropInfo = object
        paramGradient: Layer
        preActivation: seq[Float]
        postActivation: seq[Float]
        inputGradient: seq[Float]
    BackpropInfo = object
        layers: seq[LayerBackpropInfo]
        input: seq[Float]
        numSummedGradients: int

func setZero(s: var seq[Float]) =
    for v in s.mitems:
        v = 0

func setZero*(backpropInfo: var BackpropInfo) =
    for layerBackpropInfo in backpropInfo.layers.mitems:
        layerBackpropInfo.paramGradient.bias.setZero
        layerBackpropInfo.paramGradient.weights.setZero
        layerBackpropInfo.inputGradient.setZero
    backpropInfo.numSummedGradients = 0

func getBackpropInfo*(network: Network): BackpropInfo =
    for layer in network.layers:
        result.layers.add(LayerBackpropInfo(
            paramGradient: layer,
            inputGradient: newSeq[Float](layer.numInputs)
        ))
    result.setZero

func addGradient*(network: var Network, backpropInfo: BackpropInfo, lr: Float) =
    assert network.layers.len == backpropInfo.layers.len

    for layerIndex in 0..<network.layers.len:
        assert network.layers[layerIndex].bias.len == backpropInfo.layers[layerIndex].paramGradient.bias.len
        assert network.layers[layerIndex].weights.len == backpropInfo.layers[layerIndex].paramGradient.weights.len
        
        for i in 0..<network.layers[layerIndex].bias.len:
            network.layers[layerIndex].bias[i] +=
                (lr * backpropInfo.layers[layerIndex].paramGradient.bias[i]) / backpropInfo.numSummedGradients.Float
        
        for i in 0..<network.layers[layerIndex].weights.len:
            network.layers[layerIndex].weights[i] +=
                (lr * backpropInfo.layers[layerIndex].paramGradient.weights[i]) / backpropInfo.numSummedGradients.Float

func newLayer*(numInputs, numOutputs: int, activation: ActivationFunction): Layer =
    assert numInputs > 0 and numOutputs > 0

    Layer(
        bias: newSeq[Float](numOutputs),
        weights: newSeq[Float](numInputs * numOutputs),
        numInputs: numInputs,
        numOutputs: numOutputs,
        activation: activation
    )

func newNetwork*(numInputs: int, layers: varargs[tuple[numNeurons: int, activation: ActivationFunction]]): Network =
    assert layers.len >= 2, "Network needs at least one input layer and one output layer"

    for i in 0..<layers.len:
        result.layers.add(newLayer(
            numInputs = if i == 0: numInputs else: layers[i-1].numNeurons,
            numOutputs = layers[i].numNeurons,
            activation = layers[i].activation
        ))

func `$`*(network: Network): string =
    if network.layers.len == 0:
        return
    result &= $network.layers[0].numInputs
    for layer in network.layers:
        result &= "\n" & $layer.numOutputs & " " & layer.activation.name

func weightIndex(inNeuron, outNeuron, numInputs, numOutputs: int): int =
    assert inNeuron in 0..<numInputs
    assert outNeuron in 0..<numOutputs

    outNeuron + numOutputs * inNeuron;

func forward(
    layer: Layer,
    input: openArray[Float],
    layerBackpropInfo: var (Nothing or LayerBackpropInfo)
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

    assert layerBackpropInfo.paramGradient.numOutputs == layer.numOutputs
    assert layerBackpropInfo.paramGradient.bias.len == layer.numOutputs
    assert layerBackpropInfo.paramGradient.numInputs == layer.numInputs
    assert layerBackpropInfo.paramGradient.weights.len == layer.numInputs * layer.numOutputs
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
        layerBackpropInfo.paramGradient.bias[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]


    for inNeuron in 0..<layer.numInputs:
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            layerBackpropInfo.paramGradient.weights[i] +=
                inPostActivation[inNeuron] * layerBackpropInfo.paramGradient.bias[outNeuron]
            layerBackpropInfo.inputGradient[inNeuron] +=
                layer.weights[i] * layerBackpropInfo.paramGradient.bias[outNeuron]

func forwardInternal(
    network: Network,
    input: openArray[Float],
    backpropInfo: var (BackpropInfo or Nothing)
): seq[Float] =

    result = newSeq[Float](network.layers[^1].numOutputs)

    assert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    assert input.len == network.layers[0].numInputs, "Input size and input size of first layer must be the same"
    assert result.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"

    when backpropInfo isnot Nothing:
        assert backpropInfo.layers.len == network.layers.len

        backpropInfo.input = input.toSeq

    for i in 0..<network.layers.len:
        result = forward(
            layer = network.layers[i],
            input = if i == 0: input else: result,
            layerBackpropInfo = when backpropInfo is Nothing: backpropInfo else: backpropInfo.layers[i]
        )

func forward*(
    network: Network,
    input: openArray[Float],
    backpropInfo: var BackpropInfo
): seq[Float] =
    network.forwardInternal(input, backpropInfo)

func forward*(
    network: Network,
    input: openArray[Float]
): seq[Float] =
    var nothing: Nothing
    network.forwardInternal(input, nothing)

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
            layer = network.layers[i],
            outGradient = if i == network.layers.len - 1: lossGradient else: backpropInfo.layers[i + 1].inputGradient,
            inPostActivation = if i == 0: backpropInfo.input else: backpropInfo.layers[i - 1].postActivation,
            layerBackpropInfo = backpropInfo.layers[i]
        )

    backpropInfo.numSummedGradients += 1
    

#[-----------------Activation Functions-----------------]#


const sigmoid* = ActivationFunction(
    f: proc(x: Float): Float = 1.Float / (1.Float + exp(-x)),
    df: proc(x: Float): Float =
        let t = 1.Float / (1.Float + exp(-x))
        t * (1.Float - t),
    name: "sigmoid"
)

const softplus* = ActivationFunction(
    f: proc(x: Float): Float = ln(1.Float + exp(x)),
    df: proc(x: Float): Float = 1.Float / (1.Float + exp(-x)),
    name: "softplus"
)

const silu* = ActivationFunction(
    f: proc(x: Float): Float = x / (1.Float + exp(-x)),
    df: proc(x: Float): Float = (1.Float + exp(-x) + x * exp(-x)) / pow(1.Float + exp(-x), 2.Float),
    name: "silu"
)

const relu* = ActivationFunction(
    f: proc(x: Float): Float = (if x > 0: x else: 0.Float),
    df: proc(x: Float): Float = (if x > 0: 1.Float else: 0.Float),
    name: "relu"
)

const elu* = ActivationFunction(
    f: proc(x: Float): Float = (if x > 0: x else: exp(x) - 1.Float),
    df: proc(x: Float): Float = (if x > 0: 1.Float else: exp(x)),
    name: "elu"
)

const leakyRelu* = ActivationFunction(
    f: proc(x: Float): Float = (if x > 0: x else: 0.01.Float * x),
    df: proc(x: Float): Float = (if x > 0: 1.Float else: 0.01.Float),
    name: "leakyRelu"
)

const identity* = ActivationFunction(
    f: proc(x: Float): Float = x,
    df: proc(x: Float): Float = 1.Float,
    name: "identity"
)

const tanh* = ActivationFunction(
    f: proc(x: Float): Float = tanh(x),
    df: proc(x: Float): Float = 1.Float - pow(tanh(x), 2.Float),
    name: "tanh"
)

func mseGradient*(
    target: openArray[Float],
    output: openArray[Float]
): seq[Float] =
    result = newSeq[Float](target.len)

    assert target.len == output.len
    assert output.len == result.len

    for i in 0..<target.len:
        result[i] = 2.Float * (output[i] - target[i])



var model = newNetwork(1000, (256, relu), (256, relu), (1, sigmoid))
echo model

var
    input = newSeq[Float](1000)
    backpropInfo = model.getBackpropInfo

let x = model.forward(input, backpropInfo)
echo x
for i in 0..1000:
    backpropInfo.setZero
    model.backward(x.mseGradient(@[1.0'f32]), backpropInfo)
    model.addGradient(backpropInfo, 0.01'f32)
echo model.forward(input)