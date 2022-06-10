import
    math,
    sequtils,
    random,
    fenv,
    marshal,
    tables,
    locks

type
    Float* = float32
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
        sparseInput: seq[tuple[index: int, value: Float]]
        numSummedGradients: int

var
    activationFunctions: Table[string, ActivationFunction]
    mutexAddActivationFunctions: Lock

func newActivationFunction*(
        f: proc(x: Float): Float {.noSideEffect.},
        df: proc(x: Float): Float {.noSideEffect.},
        name: string
): ActivationFunction =
    result = ActivationFunction(f: f, df: df, name: name)
    {.cast(noSideEffect).}:
        withLock mutexAddActivationFunctions:
            doAssert not activationFunctions.hasKey(name), "Each activation function needs to have a unique name "
            activationFunctions[name] = result

let sigmoid* = newActivationFunction(
    f = proc(x: Float): Float = 1.Float / (1.Float + exp(-x)),
    df = proc(x: Float): Float =
        let t = 1.Float / (1.Float + exp(-x))
        t * (1.Float - t),
    name = "sigmoid"
)

let softplus* = newActivationFunction(
    f = proc(x: Float): Float = ln(1.Float + exp(x)),
    df = proc(x: Float): Float = 1.Float / (1.Float + exp(-x)),
    name = "softplus"
)

let silu* = newActivationFunction(
    f = proc(x: Float): Float = x / (1.Float + exp(-x)),
    df = proc(x: Float): Float = (1.Float + exp(-x) + x * exp(-x)) / pow(1.Float + exp(-x), 2.Float),
    name = "silu"
)

let relu* = newActivationFunction(
    f = proc(x: Float): Float = (if x > 0: x else: 0.Float),
    df = proc(x: Float): Float = (if x > 0: 1.Float else: 0.Float),
    name = "relu"
)

let elu* = newActivationFunction(
    f = proc(x: Float): Float = (if x > 0: x else: exp(x) - 1.Float),
    df = proc(x: Float): Float = (if x > 0: 1.Float else: exp(x)),
    name = "elu"
)

func leakyRelu*(a: Float = 0.01): ActivationFunction = newActivationFunction(
    f = proc(x: Float): Float = (if x > 0: x else: a * x),
    df = proc(x: Float): Float = (if x > 0: 1.Float else: a),
    name = "leakyRelu"
)

let identity* = newActivationFunction(
    f = proc(x: Float): Float = x,
    df = proc(x: Float): Float = 1.Float,
    name = "identity"
)

let tanh* = newActivationFunction(
    f = proc(x: Float): Float = tanh(x),
    df = proc(x: Float): Float = 1.Float - pow(tanh(x), 2.Float),
    name = "tanh"
)

func mseGradient*(
    target: openArray[Float],
    output: openArray[Float]
): seq[Float] =
    result = newSeq[Float](target.len)

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len
    assert output.len == result.len

    for i in 0..<target.len:
        result[i] = 2.Float * (output[i] - target[i])

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

proc generateGaussianNoise(mu: Float = 0.Float, sigma: Float = 1.Float): Float =
    # Generates values from a normal distribution.
    # Translated from https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Implementation.
    var u1: Float
    var u2: Float
    while true:
        u1 = rand(1.Float)
        u2 = rand(1.Float)
        if u1 > epsilon(Float): break
    let mag: Float = sigma * sqrt(-2 * ln(u1))
    mag * cos(2 * PI * u2) + mu

func initKaimingNormal*(network: var Network) =
    for layer in network.layers.mitems:
        layer.bias = newSeq[Float](layer.bias.len) # set bias to zero

        let std = sqrt(2.Float / layer.numInputs.Float)

        for v in layer.weights.mitems:
            {.cast(noSideEffect).}:
                v = generateGaussianNoise(mu = 0.Float, sigma = std)
    

func newNetwork*(input: int, layers: varargs[tuple[numNeurons: int, activation: ActivationFunction]]): Network =
    doAssert layers.len >= 1, "Network needs at least one layer"

    for i in 0..<layers.len:
        result.layers.add(newLayer(
            numInputs = if i == 0: input else: layers[i-1].numNeurons,
            numOutputs = layers[i].numNeurons,
            activation = layers[i].activation
        ))

    result.initKaimingNormal()

func toJson*(network: Network): string =
    {.cast(noSideEffect).}:
        $$network

func toNetwork*(json: string): Network =
    {.cast(noSideEffect).}:
        result = to[Network](json)
        withLock mutexAddActivationFunctions:
            for layer in result.layers.mitems:
                doAssert(
                    activationFunctions.hasKey(layer.activation.name),
                    "Activation function '" & layer.activation.name & "' must be created using 'newActivationFunction'"
                )
                layer.activation = activationFunctions[layer.activation.name]


func `$`*(network: Network): string =
    if network.layers.len == 0:
        return
    result &= $network.layers[0].numInputs
    for layer in network.layers:
        result &= "\n" & $layer.numOutputs & " " & layer.activation.name

func weightIndex(inNeuron, outNeuron, numInputs, numOutputs: int): int =
    outNeuron + numOutputs * inNeuron;

func feedForwardLayer(
    layer: Layer,
    input: openArray[Float],
    layerBackpropInfo: var (Nothing or LayerBackpropInfo)
): seq[Float] =

    assert input.len == layer.numInputs
    assert layer.bias.len == layer.numOutputs
    assert layer.weights.len == layer.numInputs * layer.numOutputs

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

func feedForwardLayer(
    layer: Layer,
    input: openArray[tuple[index: int, value: Float]],
    layerBackpropInfo: var (Nothing or LayerBackpropInfo)
): seq[Float] =

    assert layer.bias.len == layer.numOutputs
    assert layer.weights.len == layer.numInputs * layer.numOutputs

    result = layer.bias

    for (inNeuron, value) in input:
        assert inNeuron in 0..<layer.numInputs
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            result[outNeuron] += layer.weights[i] * value

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.preActivation = result

    for value in result.mitems:
        value = layer.activation.f(value)

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.postActivation = result

{.emit: ["""
#include <omp.h>
"""].}

func backPropagateLayer(
    layer: Layer,
    outGradient: openArray[Float],
    inPostActivation: openArray[Float],
    layerBackpropInfo: var LayerBackpropInfo,
    calcInGradient: bool = true
) =
    assert layerBackpropInfo.paramGradient.numOutputs == layer.numOutputs
    assert layerBackpropInfo.paramGradient.bias.len == layer.numOutputs
    assert layerBackpropInfo.paramGradient.numInputs == layer.numInputs
    assert layerBackpropInfo.paramGradient.weights.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert inPostActivation.len == layer.numInputs
    assert layerBackpropInfo.inputGradient.len == layer.numInputs

    for inNeuron in 0..<layer.numInputs:
        layerBackpropInfo.inputGradient[inNeuron] = 0

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.paramGradient.bias[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]

    # for inNeuron in 0..<layer.numInputs:
    #     for outNeuron in 0..<layer.numOutputs:
    #         let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
    #         layerBackpropInfo.paramGradient.weights[i] +=
    #             inPostActivation[inNeuron] * layerBackpropInfo.paramGradient.bias[outNeuron]
    #         layerBackpropInfo.inputGradient[inNeuron] +=
    #             layer.weights[i] * layerBackpropInfo.paramGradient.bias[outNeuron]
    # ^this is implemented below using OpenMP to utilize SIMD instructions
    {.emit: ["""
    for(size_t inNeuron = 0; inNeuron < """, layer.numInputs, """; ++inNeuron){
        """, Float, """* in_neurons_gradient_in = """, layerBackpropInfo.inputGradient, """.p->data;
        #pragma omp simd reduction(+ : in_neurons_gradient_in[inNeuron])
        for(size_t outNeuron = 0; outNeuron < """, layer.numOutputs, """; ++outNeuron){
            const size_t i = """, weightIndex, """(inNeuron, outNeuron, """, layer.numInputs, """,""", layer.numOutputs, """);
            """, layerBackpropInfo.paramGradient.weights, """.p->data[i] +=
                """, inPostActivation, """[inNeuron] * """, layerBackpropInfo.paramGradient.bias, """.p->data[outNeuron];
            if(""",calcInGradient,""")
            in_neurons_gradient_in[inNeuron] +=
                """, layer.weights, """.p->data[i] * """, layerBackpropInfo.paramGradient.bias, """.p->data[outNeuron];
        }
    }    
    """].}

func backPropagateLayer(
    layer: Layer,
    outGradient: openArray[Float],
    inPostActivation: openArray[tuple[index: int, value: Float]],
    layerBackpropInfo: var LayerBackpropInfo,
    calcInGradient: bool = true
) =
    assert layerBackpropInfo.paramGradient.numOutputs == layer.numOutputs
    assert layerBackpropInfo.paramGradient.bias.len == layer.numOutputs
    assert layerBackpropInfo.paramGradient.numInputs == layer.numInputs
    assert layerBackpropInfo.paramGradient.weights.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert inPostActivation.len == layer.numInputs or inPostActivation is openArray[tuple[index: int, value: Float]]
    assert layerBackpropInfo.inputGradient.len == layer.numInputs

    for inNeuron in 0..<layer.numInputs:
        layerBackpropInfo.inputGradient[inNeuron] = 0

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.paramGradient.bias[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]

    for (inNeuron, value) in inPostActivation:
        assert inNeuron in 0..<layer.numInputs
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            layerBackpropInfo.paramGradient.weights[i] +=
                value * layerBackpropInfo.paramGradient.bias[outNeuron]

    if calcInGradient:
        for inNeuron in 0..<layer.numInputs:
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.inputGradient[inNeuron] +=
                    layer.weights[i] * layerBackpropInfo.paramGradient.bias[outNeuron]

func forwardInternal(
    network: Network,
    input: openArray[Float] or openArray[tuple[index: int, value: Float]],
    backpropInfo: var (BackpropInfo or Nothing)
): seq[Float] =

    result = newSeq[Float](network.layers[^1].numOutputs)

    doAssert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    doAssert result.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"
    doAssert(
        input.len == network.layers[0].numInputs or input isnot openArray[Float],
        "Input size and input size of first layer must be the same"
    )

    when backpropInfo isnot Nothing:
        assert backpropInfo.layers.len == network.layers.len
        when input is openArray[Float]:
            backpropInfo.input = input.toSeq
        else:
            backpropInfo.sparseInput = input.toSeq


    result = feedForwardLayer(
        layer = network.layers[0],
        input = input,
        layerBackpropInfo = when backpropInfo is Nothing: backpropInfo else: backpropInfo.layers[0]
    )

    for i in 1..<network.layers.len:
        result = feedForwardLayer(
            layer = network.layers[i],
            input = result,
            layerBackpropInfo = when backpropInfo is Nothing: backpropInfo else: backpropInfo.layers[i]
        )

func forward*(network: Network, input: openArray[Float], backpropInfo: var BackpropInfo): seq[Float] =
    network.forwardInternal(input, backpropInfo)

func forward*(network: Network, input: openArray[Float]): seq[Float] =
    var nothing: Nothing
    network.forwardInternal(input, nothing)

func forward*(network: Network, input: openArray[tuple[index: int, value: Float]], backpropInfo: var BackpropInfo): seq[Float] =
    network.forwardInternal(input, backpropInfo)

func forward*(network: Network, input: openArray[tuple[index: int, value: Float]]): seq[Float] =
    var nothing: Nothing
    network.forwardInternal(input, nothing)

func backward*(
    network: Network,
    lossGradient: openArray[Float],
    backpropInfo: var BackpropInfo
) =
    doAssert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    doAssert lossGradient.len == network.layers[^1].numOutputs, "Loss size and output size of last layer must be the same"
    doAssert backpropInfo.layers.len == network.layers.len

    for i in countdown(network.layers.len - 1, 1):
        backPropagateLayer(
            layer = network.layers[i],
            outGradient = if i == network.layers.len - 1: lossGradient else: backpropInfo.layers[i + 1].inputGradient,
            inPostActivation = backpropInfo.layers[i - 1].postActivation,
            layerBackpropInfo = backpropInfo.layers[i]
        )

    template propagateLast(input: auto) =
        backPropagateLayer(
            layer = network.layers[0],
            outGradient = backpropInfo.layers[1].inputGradient,
            inPostActivation = input,
            layerBackpropInfo = backpropInfo.layers[0],
            calcInGradient = false
        )

    if backpropInfo.input.len > 0:
        propagateLast(backpropInfo.input)
    else:
        doAssert backpropInfo.sparseInput.len > 0, "BackpropInfo is not set up correctly for backpropagation"
        propagateLast(backpropInfo.sparseInput)

    backpropInfo.numSummedGradients += 1


# when isMainModule:
#     var model = newNetwork(10, (2, relu))
#     let s = $$model
#     var newModel = to[Network](s)
#     echo newModel

    # var model = newNetwork(1000, (256, relu), (256, relu), (1, sigmoid))
    # echo model

    # var
    #     input = newSeq[Float](1000)
    #     backpropInfo = model.getBackpropInfo

    # let x = model.forward(input, backpropInfo)
    # echo x
    # for i in 0..1000:
    #     backpropInfo.setZero
    #     model.backward(x.mseGradient(@[1.0'f32]), backpropInfo)
    #     model.addGradient(backpropInfo, 0.01'f32)
    # echo model.forward(input)