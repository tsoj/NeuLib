import std/[
    math,
    sequtils,
    random,
    json,
    strformat,
    streams
]

type
    ActivationFunction* = enum
        identity, relu, leakyRelu, sigmoid, tanh
    Layer* = object
        bias*: seq[float32]
        weights*: seq[float32]
        numInputs*: int
        numOutputs*: int
        activation*: ActivationFunction
    Network* = object
        layers*: seq[Layer]
    SparseElement* = tuple[index: int, value: float32]
    SparseOneElement* = tuple[index: int]
    LayerBackpropInfo = object
        preActivation: seq[float32]
        postActivation: seq[float32]
        biasGradient: seq[float32]
        weightsGradient: seq[float32]
        inputGradient: seq[float32]
    InputType = enum
        full, sparse, sparseOnes
    NetworkInput = object
        case isSparseInput: InputType
        of sparse: sparseInput: seq[SparseElement]
        of sparseOnes: sparseOnesInput: seq[SparseOneElement]
        of full: input: seq[float32]
    BackpropInfo* = object
        layers: seq[LayerBackpropInfo]
        numSummedGradients: int
        input: NetworkInput
    Moments* = object
        lr: float32
        beta: float32
        values: seq[Layer]
    Nothing = tuple[]

#----------- Utility Functions -----------#

func toSparse*(input: openArray[float32], margin: float32 = 0.01): seq[SparseElement] =
    ## Returns a sparse representation of a given array. Elements that are closer to zero than `margin` will not
    ## be added to the result.

    for i, a in input.pairs:
        if abs(a) >= margin:
            result.add((i, a))

func toSparseOnes*(input: openArray[float32], splitAt: float32): seq[SparseOneElement] =
    ## Returns a sparse ones representation of a given array. Elements that are bigger than or equal to `splitAt`
    ## will be counted as ones, all other valus as zeros

    for i, a in input.pairs:
        if a >= splitAt:
            result.add((index: i))

func apply(x: var openArray[float32], y: openArray[float32], op: proc(x: var float32, y: float32) {.noSideEffect.}) =
    doAssert x.len == y.len
    for i in 0..<x.len:
        op(x[i], y[i])

func `+=`*(x: var openArray[float32], y: openArray[float32]) =
    apply(x,y, proc(x: var float32, y: float32) = x += y)
func `-=`*(x: var openArray[float32], y: openArray[float32]) =
    apply(x,y, proc(x: var float32, y: float32) = x -= y)
func `*=`*(x: var openArray[float32], y: openArray[float32]) =
    apply(x,y, proc(x: var float32, y: float32) = x *= y)
func `/=`*(x: var openArray[float32], y: openArray[float32]) =
    apply(x,y, proc(x: var float32, y: float32) = x /= y)

func `+=`*(x: var openArray[float32], y: float32) =
    apply(x, proc(x: float32): float32 = x + y)
func `-=`*(x: var openArray[float32], y: float32) =
    apply(x, proc(x: float32): float32 = x - y)
func `*=`*(x: var openArray[float32], y: float32) =
    apply(x, proc(x: float32): float32 = x * y)
func `/=`*(x: var openArray[float32], y: float32) =
    apply(x, proc(x: float32): float32 = x / y)

func `+`*(x, y: openArray[float32]): seq[float32] =
    result &= x
    result += y
func `-`*(x, y: openArray[float32]): seq[float32] =
    result &= x
    result -= y
func `*`*(x, y: openArray[float32]): seq[float32] =
    result &= x
    result *= y
func `/`*(x, y: openArray[float32]): seq[float32] =
    result &= x
    result /= y

func `+`*(x: openArray[float32], y: float32): seq[float32] =
    result &= x
    result += y
func `-`*(x: openArray[float32], y: float32): seq[float32] =
    result &= x
    result -= y
func `+`*(x: float32, y: openArray[float32]): seq[float32] =
    y + x
func `-`*(x: float32, y: openArray[float32]): seq[float32] =
    result &= y
    apply(result, proc(a: float32): float32 = x - a)

func `*`*(x: openArray[float32], y: float32): seq[float32] =
    result &= x
    result *= y
func `/`*(x: openArray[float32], y: float32): seq[float32] =
    result &= x
    result /= y
func `*`*(x: float32, y: openArray[float32]): seq[float32] =
    y * x
func `/`*(x: float32, y: openArray[float32]): seq[float32] =
    result &= y
    apply(result, proc(a: float32): float32 = x/a)

#----------- Activation and Loss Functions -----------#

func getActivationFunction*(activation: ActivationFunction): auto =
    case activation:
    of identity: return proc(x: float32): float32 = x
    of relu: return proc(x: float32): float32 = (if x > 0: x else: 0)
    of leakyRelu: return proc(x: float32): float32 = (if x > 0: x else: 0.01f * x)
    of sigmoid: return proc(x: float32): float32 = (1.0f / (1.0f + exp(-x)))
    of tanh: return proc(x: float32): float32 = (x.tanh)

func getActivationFunctionDerivative*(activation: ActivationFunction): auto =
    case activation:
    of identity: return proc(x: float32): float32 = 1.0f
    of relu: return proc(x: float32): float32 = (if x > 0: 1 else: 0)
    of leakyRelu: return proc(x: float32): float32 = (if x > 0: 1 else: 0.01f)
    of sigmoid: return proc(x: float32): float32 =
        let t = 1.0f / (1.0f + exp(-x))
        t * (1.0f - t)
    of tanh: return proc(x: float32): float32 = 1.0f - pow(x.tanh, 2.0f)

func customLossGradient*[TIn](
    target: openArray[TIn],
    output: openArray[float32],
    calculateElementLossGradient: proc(t: TIn, o: float32): float32 {.noSideEffect.}
): seq[float32] =
    ## Returns a custom loss gradient.
    ## `target` is the desired output vector, `output` is the actual output.

    result = newSeq[float32](target.len)

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len
    assert output.len == result.len

    for i in 0..<target.len:
        result[i] = calculateElementLossGradient(target[i], output[i])

func customLoss*[TIn](
    target: openArray[TIn],
    output: openArray[float32],
    calculateElementLoss: proc(t: TIn, o: float32): float32 {.noSideEffect.}
): float32 =
    ## Returns a custom loss value.
    ## `target` is the desired output vector, `output` is the actual output.

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len

    for i in 0..<target.len:
        result += calculateElementLoss(target[i], output[i])
    result /= target.len.float32

func mseGradient*(
    target: openArray[float32],
    output: openArray[float32]
): seq[float32] =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLossGradient(target, output, proc(t, o: float32): float32 = 2.0f * (o - t))

func mseLoss*(
    target: openArray[float32],
    output: openArray[float32]
): float32 =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLoss(target, output, proc(t, o: float32): float32 = (o - t)*(o - t))

#----------- Network and Gradient Stuff  -----------#

func setZero(s: var seq[SomeNumber]) =
    for v in s.mitems:
        v = 0

func setZero*(backpropInfo: var BackpropInfo) =
    ## Resets the gradient stored in `backpropInfo` to zero.

    for layerBackpropInfo in backpropInfo.layers.mitems:
        layerBackpropInfo.biasGradient.setZero
        layerBackpropInfo.weightsGradient.setZero
        layerBackpropInfo.inputGradient.setZero
    backpropInfo.numSummedGradients = 0

func newBackpropInfo*(network: Network): BackpropInfo =
    ## Creates `BackpropInfo` that can be used to do backwards passes with `network`.

    for layer in network.layers:
        result.layers.add(LayerBackpropInfo(
            biasGradient: layer.bias,
            weightsGradient: layer.weights,
            inputGradient: newSeq[float32](layer.numInputs)
        ))
    result.setZero

func inputGradient*(backpropInfo: BackpropInfo): seq[float32] =
    ## Returns the input gradient that has been calculated during a backward pass (
    ## when calling `backward <#backward,Network,openArray[float32],BackpropInfo,staticbool>`_,
    ## the parameter `calculateInputGradient` must be set to `true`)

    doAssert backpropInfo.layers.len >= 1, "BackpropInfo needs at least one input layer and one output layer"
    backpropInfo.layers[0].inputGradient

func newMoments*(network: Network, lr = 0.1, beta = 0.9): Moments =
    ## Creates `Moments` that can be used for using a gradient to optimize `network`

    for layer in network.layers:
        result.values.add layer
        result.values[^1].bias.setZero
        result.values[^1].weights.setZero

    result.beta = beta
    result.lr = lr

func addGradient*(network: var Network, backpropInfo: BackpropInfo, lr: float32) =
    ## Applies the gradient accumulated during backwards passes in `backpropInfo` to `network`.
    ## Learning rate can be specified with `lr`.

    assert network.layers.len == backpropInfo.layers.len

    for layerIndex in 0..<network.layers.len:
        assert network.layers[layerIndex].bias.len == backpropInfo.layers[layerIndex].biasGradient.len
        assert network.layers[layerIndex].weights.len == backpropInfo.layers[layerIndex].weightsGradient.len
        
        network.layers[layerIndex].bias -= (lr / backpropInfo.numSummedGradients.float32) * backpropInfo.layers[layerIndex].biasGradient

        network.layers[layerIndex].weights -= (lr / backpropInfo.numSummedGradients.float32) * backpropInfo.layers[layerIndex].weightsGradient

func addGradient*(network: var Network, backpropInfo: BackpropInfo, moments: var Moments) =
    ## Applies the gradient accumulated during backwards passes in `backpropInfo` to `network`.
    ## Uses moments.

    assert network.layers.len == backpropInfo.layers.len
    assert network.layers.len == moments.values.len
    
    for layerIndex in 0..<network.layers.len:
        assert network.layers[layerIndex].bias.len == backpropInfo.layers[layerIndex].biasGradient.len
        assert network.layers[layerIndex].weights.len == backpropInfo.layers[layerIndex].weightsGradient.len

        assert network.layers[layerIndex].bias.len == moments.values[layerIndex].bias.len
        assert network.layers[layerIndex].weights.len == moments.values[layerIndex].weights.len

        moments.values[layerIndex].bias *= moments.beta
        moments.values[layerIndex].bias += 
            ((1 - moments.beta) / backpropInfo.numSummedGradients.float32) *
            backpropInfo.layers[layerIndex].biasGradient
        network.layers[layerIndex].bias -= moments.lr * moments.values[layerIndex].bias

        moments.values[layerIndex].weights *= moments.beta
        moments.values[layerIndex].weights += 
            ((1 - moments.beta) / backpropInfo.numSummedGradients.float32) *
            backpropInfo.layers[layerIndex].weightsGradient
        network.layers[layerIndex].weights -= moments.lr * moments.values[layerIndex].weights

func newLayer(numInputs, numOutputs: int, activation: ActivationFunction): Layer =
    assert numInputs > 0 and numOutputs > 0

    Layer(
        bias: newSeq[float32](numOutputs),
        weights: newSeq[float32](numInputs * numOutputs),
        numInputs: numInputs,
        numOutputs: numOutputs,
        activation: activation
    )

func initKaimingNormal*(network: var Network, randState: var Rand) =
    ## Randomly initializes the parameters of `network` using a method described in
    ## `"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    ## <https://arxiv.org/abs/1502.01852>`_.

    for layer in network.layers.mitems:
        layer.bias = newSeq[float32](layer.bias.len) # set bias to zero

        let std = sqrt(2.float32 / layer.numInputs.float32)

        for v in layer.weights.mitems:
            v = randState.gauss(mu = 0.0, sigma = std.float).float32

func initKaimingNormal*(network: var Network, seed: int64 = 0) =
    ## Randomly initializes the parameters of `network` using a method described in
    ## `"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    ## <https://arxiv.org/abs/1502.01852>`_.

    var randState: Rand = initRand(seed)
    network.initKaimingNormal(randState)

func newNetwork*(
    input: int,
    layers: varargs[tuple[numNeurons: int, activation: ActivationFunction]]
): Network =
    ## Creates new instance of `Network`.
    ## `input` is the number of input units.
    ## `layers` describes how many neurons the following layers have and which activation function they use.
    ## The model will be randomly initialized using Kaiming initialization.
    ## 
    ## Example:
    ##
    ## .. code-block:: Nim
    ##
    ##   var model = newNetwork(
    ##     784,
    ##     (40, relu),
    ##     (40, relu),
    ##     (10, sigmoid)
    ##   )
    ## 
    ## Describes a model that looks like this:
    ## 
    ## .. code-block::
    ## 
    ##   input:  784 neurons
    ##                \/ 
    ##   hidden:  40 neurons -> relu
    ##                ||
    ##   hidden:  40 neurons -> relu
    ##                \/
    ##   output:  10 neurons -> sigmoid

    doAssert layers.len >= 1, "Network needs at least one layer"

    for i in 0..<layers.len:
        result.layers.add(newLayer(
            numInputs = if i == 0: input else: layers[i-1].numNeurons,
            numOutputs = layers[i].numNeurons,
            activation = layers[i].activation
        ))

    result.initKaimingNormal(seed = 0)

func description*(network: Network): string =
    ## Returns a string conveying some basic information about the structure  of the network.

    if network.layers.len == 0:
        return "Empty network"

    var maxLength = len($network.layers[0].numInputs)
    for layer in network.layers:
        maxLength = max(maxLength, len($layer.numOutputs))
    
    func getNumberString(num: int): string =
        result = $num
        while result.len < maxLength:
            result = " " & result

    let spacesBeforeWeights = block:
        var s = ""
        for i in 0..<(maxLength + "neurons".len) div 2:
            s &= " "
        s

    result &= "Input:  " & getNumberString(network.layers[0].numInputs) & " neurons\n"

    for i, layer in network.layers.pairs:
        result &= "        "
        result &= spacesBeforeWeights & (
            if layer.numOutputs > layer.numInputs: "/\\\n"
            elif layer.numOutputs < layer.numInputs: "\\/\n"
            else: "||\n"
        )
        let isLast = i == network.layers.len - 1
        if isLast:
            result &= "Output: "
        else:
            result &= "Hidden: "

        result &= getNumberString(layer.numOutputs) & " neurons -> " & $layer.activation
        
        if not isLast:
            result &= "\n"

func toJsonString*(network: Network): string =
    ## Returns the JSON represenation of `network`.

    {.cast(noSideEffect).}:
        let a = %*(network)
        pretty(a)

func toNetwork*(jsonString: string): Network =
    ## Creates a `Network` from a JSON formatted string representation.

    {.cast(noSideEffect).}:
        let jsonNode = jsonString.parseJson
    result = jsonNode.to Network

proc writeSeq[float32](stream: Stream, x: seq[float32], writeProc: proc(s: Stream, x: float32)) =
    stream.write x.len.int64
    for a in x:
        stream.writeProc a
    
proc readSeq[float32](stream: Stream, r: var seq[float32], readProc: proc(s: Stream, x: var float32)) =
    let len = stream.readInt64
    for i in 0..<len:
        var element: float32
        stream.readProc element
        r.add element

proc writeLayer(stream: Stream, layer: Layer) =
    stream.writeSeq layer.bias, write
    stream.writeSeq layer.weights, write
    stream.write layer.numInputs.int64
    stream.write layer.numOutputs.int64
    stream.write layer.activation.int64
    
proc readLayer(stream: Stream, layer: var Layer) =
    stream.readSeq layer.bias, read
    stream.readSeq layer.weights, read
    layer.numInputs = stream.readInt64
    layer.numOutputs = stream.readInt64
    layer.activation = stream.readInt64.ActivationFunction

proc writeNetwork*(stream: Stream, network: Network) =
    ## Writes the binary representation of `network` into `stream`.
    ## Can be loaded again using `readNetwork <#readNetwork,Stream>`_.

    stream.writeSeq network.layers, writeLayer

proc readNetwork*(stream: Stream): Network =
    ## Loads a network from `stream`, assuming a binary representation as created
    ## by `writeNetwork <#writeNetwork,Stream,Network>`_.

    stream.readSeq result.layers, readLayer

proc saveToFile*(network: Network, fileName: string) =
    ## Saves a binary representation of `network` into a file.
    ## Can be loaded again using `loadNetworkFromFile <#loadNetworkFromFile,string>`_.

    var fileStream = newFileStream(fileName, fmWrite)
    if fileStream.isNil:
        raise newException(IOError, "Couldn't open file: " & fileName)
    fileStream.writeNetwork network
    fileStream.close

proc loadNetworkFromFile*(fileName: string): Network =
    ## Loads a network from a binary file that has been saved 
    ## using `saveToFile <#saveToFile,Network,string>`_.

    var fileStream = newFileStream(fileName, fmRead)
    if fileStream.isNil:
        raise newException(IOError, "Couldn't open file: " & fileName)
    result = fileStream.readNetwork
    fileStream.close

func weightIndex*(inNeuron, outNeuron, numInputs, numOutputs: int): int =
    outNeuron + numOutputs * inNeuron

#----------- Feed Forward Functions -----------#

func feedForwardLayer(
    layer: Layer,
    input: openArray[float32 or SparseElement or SparseOneElement],
    layerBackpropInfo: var (Nothing or LayerBackpropInfo)
): seq[float32] =

    assert layer.bias.len == layer.numOutputs
    assert layer.weights.len == layer.numInputs * layer.numOutputs

    result = layer.bias

    when input is openArray[SparseElement]:
        for (inNeuron, value) in input:
            assert inNeuron in 0..<layer.numInputs
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                result[outNeuron] += layer.weights[i] * value
    elif input is openArray[SparseOneElement]:
        for (inNeuron) in input:
            assert inNeuron in 0..<layer.numInputs
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                result[outNeuron] += layer.weights[i]
    else:
        assert input.len == layer.numInputs
        for inNeuron in 0..<layer.numInputs:
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                result[outNeuron] += layer.weights[i] * input[inNeuron]

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.preActivation = result

    for value in result.mitems:
        value = getActivationFunction(layer.activation)(value)

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.postActivation = result

func forwardInternal(
    network: Network,
    input: openArray[float32 or SparseElement or SparseOneElement],
    backpropInfo: var (BackpropInfo or Nothing)
): seq[float32] =

    doAssert network.layers.len >= 1, "Network needs at least one input layer and one output layer"

    result = newSeq[float32](network.layers[^1].numOutputs)

    doAssert result.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"
    doAssert(
        input.len == network.layers[0].numInputs or input is openArray[SparseElement or SparseOneElement],
        fmt"Input size ({input.len}) and input size of first layer ({network.layers[0].numInputs}) must be the same"
    )

    when backpropInfo isnot Nothing:
        doAssert backpropInfo.layers.len == network.layers.len
        when input is openArray[SparseElement]:
            backpropInfo.input = NetworkInput(isSparseInput: sparse, sparseInput: input.toSeq)
        elif input is openArray[SparseOneElement]:
            backpropInfo.input = NetworkInput(isSparseInput: sparseOnes, sparseOnesInput: input.toSeq)
        else:
            static: doAssert input is openArray[float32]
            backpropInfo.input = NetworkInput(isSparseInput: full, input: input.toSeq)

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

func forward*(network: Network, input: openArray[float32 or SparseElement or SparseOneElement], backpropInfo: var BackpropInfo): seq[float32] =
    ## Runs the network with a given input. Also collects information in `backpropInfo` for a
    ## `backward <#backward,Network,openArray[float32],BackpropInfo,staticbool>`_ pass.
    ## Returns a `seq` with the length according to the size of the output layer of the network.
    ## 
    ## Note: When no information for a backward pass is needed,
    ## use `forward(Network, openArray[]) <#forward,Network,openArray[]>`_ for better performance.

    network.forwardInternal(input, backpropInfo)

func forward*(network: Network, input: openArray[float32 or SparseElement or SparseOneElement]): seq[float32] =
    ## Runs the network with a given input. Does not collect information for a backward pass.
    ## Returns a `seq` with the length according to the size of the output layer of the network.

    var nothing: Nothing
    network.forwardInternal(input, nothing)

func backPropagateLayer(
    layer: Layer,
    outGradient: openArray[float32],
    inPostActivation: openArray[float32 or SparseElement or SparseOneElement],
    layerBackpropInfo: var LayerBackpropInfo,
    calculateInputGradient: static bool = true
) =
    assert layerBackpropInfo.biasGradient.len == layer.numOutputs
    assert layerBackpropInfo.weightsGradient.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert layerBackpropInfo.inputGradient.len == layer.numInputs

    when calculateInputGradient:
        for inNeuron in 0..<layer.numInputs:
            layerBackpropInfo.inputGradient[inNeuron] = 0

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.biasGradient[outNeuron] =
            getActivationFunctionDerivative(layer.activation)(
                layerBackpropInfo.preActivation[outNeuron]
            ) * outGradient[outNeuron]

    when inPostActivation is openArray[SparseElement]:
        for (inNeuron, value) in inPostActivation:
            assert inNeuron in 0..<layer.numInputs
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.weightsGradient[i] +=
                    value * layerBackpropInfo.biasGradient[outNeuron]

    elif inPostActivation is openArray[SparseOneElement]:
        for (inNeuron) in inPostActivation:
            assert inNeuron in 0..<layer.numInputs
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.weightsGradient[i] += layerBackpropInfo.biasGradient[outNeuron]
    else:
        assert inPostActivation.len == layer.numInputs
        for inNeuron in 0..<layer.numInputs:
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.weightsGradient[i] +=
                    inPostActivation[inNeuron] * layerBackpropInfo.biasGradient[outNeuron]    

    when calculateInputGradient:
        for inNeuron in 0..<layer.numInputs:
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.inputGradient[inNeuron] +=
                    layer.weights[i] * layerBackpropInfo.biasGradient[outNeuron]

func backward*(
    network: Network,
    lossGradient: openArray[float32],
    backpropInfo: var BackpropInfo,
    calculateInputGradient: static bool = false
) =
    ## Calculates and adds the gradient of all network parameters in regard to the loss gradient to `backpropInfo`.
    ## When `calculateInputGradient` is set to false, the gradient for the input will not be calculated.

    doAssert network.layers.len >= 1, "Network needs at least one input layer and one output layer"
    doAssert lossGradient.len == network.layers[^1].numOutputs, "Loss size and output size of last layer must be the same"
    doAssert backpropInfo.layers.len == network.layers.len, "BackpropInfo is not set up correctly for backpropagation"

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
            outGradient = if backpropInfo.layers.len <= 1: lossGradient else: backpropInfo.layers[1].inputGradient,
            inPostActivation = input,
            layerBackpropInfo = backpropInfo.layers[0],
            calculateInputGradient = calculateInputGradient
        )

    case backpropInfo.input.isSparseInput:
    of sparse: propagateLast(backpropInfo.input.sparseInput)
    of sparseOnes: propagateLast(backpropInfo.input.sparseOnesInput)
    of full: propagateLast(backpropInfo.input.input)

    backpropInfo.numSummedGradients += 1