import std/[
    math,
    sequtils,
    random,
    json,
    tables,
    locks,
    strformat
]

import nimblas

type
    Float* = float32
    ActivationFunction* = object
        f*: proc(x: Float): Float {.noSideEffect.}
        df*: proc(x: Float): Float {.noSideEffect.}
        name*: string
    Layer* = object
        bias*: seq[Float]
        weights*: seq[Float]
        numInputs*: int
        numOutputs*: int
        activation*: ActivationFunction
    Network* = object
        ## A fully connected neural network. Contains structure and parameters.
        layers*: seq[Layer]
    SparseElement* = tuple[index: int, value: Float]
    LayerBackpropInfo* = object
        biasGradient*: seq[Float]
        weightsGradient*: seq[Float]
        preActivation: seq[Float]
        postActivation: seq[Float]
        inputGradient: seq[Float]
    BackpropInfo* = object
        layers*: seq[LayerBackpropInfo]
        input: seq[Float]
        sparseInput: seq[SparseElement]
        numSummedGradients*: int
    Moments* = object
        lr: Float
        beta: Float
        values: seq[Layer]
    Nothing = enum a

var
    activationFunctions: Table[string, ActivationFunction]
    mutexAccessActivationFunctions: Lock

#----------- Utility Functions -----------#

func toSparse*(input: openArray[Float], margin: Float = 0.01): seq[SparseElement] =
    ## Returns a sparse representation of a given array. Elements that are closer to zero than `margin` will not
    ## be added to the result.

    for i, a in input.pairs:
        if abs(a) >= margin:
            result.add((i, a))

func apply*(x: var openArray[Float], y: openArray[Float], op: proc(x: var Float, y: Float) {.noSideEffect.}) =
    doAssert x.len == y.len
    for i in 0..<x.len:
        op(x[i], y[i])

func `+=`*(x: var openArray[Float], y: openArray[Float]) =
    apply(x,y, proc(x: var Float, y: Float) = x += y)
func `-=`*(x: var openArray[Float], y: openArray[Float]) =
    apply(x,y, proc(x: var Float, y: Float) = x -= y)
func `*=`*(x: var openArray[Float], y: openArray[Float]) =
    apply(x,y, proc(x: var Float, y: Float) = x *= y)
func `/=`*(x: var openArray[Float], y: openArray[Float]) =
    apply(x,y, proc(x: var Float, y: Float) = x /= y)

func `+=`*(x: var openArray[Float], y: Float) =
    apply(x, proc(x: Float): Float = x + y)
func `-=`*(x: var openArray[Float], y: Float) =
    apply(x, proc(x: Float): Float = x - y)
func `*=`*(x: var openArray[Float], y: Float) =
    apply(x, proc(x: Float): Float = x * y)
func `/=`*(x: var openArray[Float], y: Float) =
    apply(x, proc(x: Float): Float = x / y)

func `+`*(x, y: openArray[Float]): seq[Float] =
    result &= x
    result += y
func `-`*(x, y: openArray[Float]): seq[Float] =
    result &= x
    result -= y
func `*`*(x, y: openArray[Float]): seq[Float] =
    result &= x
    result *= y
func `/`*(x, y: openArray[Float]): seq[Float] =
    result &= x
    result /= y

func `+`*(x: openArray[Float], y: Float): seq[Float] =
    result &= x
    result += y
func `-`*(x: openArray[Float], y: Float): seq[Float] =
    result &= x
    result -= y
func `+`*(x: Float, y: openArray[Float]): seq[Float] =
    y + x
func `-`*(x: Float, y: openArray[Float]): seq[Float] =
    result &= y
    apply(result, proc(a: Float): Float = x - a)

func `*`*(x: openArray[Float], y: Float): seq[Float] =
    result &= x
    result *= y
func `/`*(x: openArray[Float], y: Float): seq[Float] =
    result &= x
    result /= y
func `*`*(x: Float, y: openArray[Float]): seq[Float] =
    y * x
func `/`*(x: Float, y: openArray[Float]): seq[Float] =
    result &= y
    apply(result, proc(a: Float): Float = x/a)

#----------- Activation and Loss Functions -----------#

func newActivationFunction*(
        f: proc(x: Float): Float {.noSideEffect.},
        df: proc(x: Float): Float {.noSideEffect.},
        name: string
): ActivationFunction =
    ## Creates and registers an activation function to be used for neural network layers.
    ## `f` is the function, `df` is the derivative of that function, and `name` must be a unique name
    ## for this activation function.
    ## Cannot be run during compile time.

    result = ActivationFunction(f: f, df: df, name: name)
    {.cast(noSideEffect).}:
        withLock mutexAccessActivationFunctions:
            doAssert not activationFunctions.hasKey(name), "Each activation function needs to have a unique name "
            activationFunctions[name] = result

func getActivationFunction*(name: string): ActivationFunction =
    ## Return the activation function registered under `name`.

    {.cast(noSideEffect).}:
        withLock mutexAccessActivationFunctions:
            doAssert(
                activationFunctions.hasKey(name),
                "Activation function '" & name & "' must be created using 'newActivationFunction'"
            )
            return activationFunctions[name]


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

let leakyRelu* = newActivationFunction(
    f = proc(x: Float): Float = (if x > 0: x else: 0.01.Float * x),
    df = proc(x: Float): Float = (if x > 0: 1.Float else: 0.01.Float),
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

func customLossGradient*[T](
    target: openArray[T],
    output: openArray[Float],
    calculateElementLossGradient: proc(t: T, o: Float): Float {.noSideEffect.}
): seq[Float] =
    ## Returns a custom loss gradient.
    ## `target` is the desired output vector, `output` is the actual output.

    result = newSeq[Float](target.len)

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len
    assert output.len == result.len

    for i in 0..<target.len:
        result[i] = calculateElementLossGradient(target[i], output[i])

func customLoss*[T](
    target: openArray[T],
    output: openArray[Float],
    calculateElementLoss: proc(t: T, o: Float): Float {.noSideEffect.}
): Float =
    ## Returns a custom loss value.
    ## `target` is the desired output vector, `output` is the actual output.

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len

    for i in 0..<target.len:
        result += calculateElementLoss(target[i], output[i])
    result /= target.len.Float

func mseGradient*(
    target: openArray[Float],
    output: openArray[Float]
): seq[Float] =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLossGradient(target, output, proc(t, o: Float): Float = 2.Float * (o - t))

func mseLoss*(
    target: openArray[Float],
    output: openArray[Float]
): Float =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLoss(target, output, proc(t, o: Float): Float = (o - t)*(o - t))


#----------- Network and Gradient Stuff  -----------#

func setZero(s: var seq[Float]) =
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
            inputGradient: newSeq[Float](layer.numInputs)
        ))
    result.setZero

func newMoments*(network: Network, lr = 0.1, beta = 0.9): Moments =
    ## Creates `Moments` that can be used for using a gradient to optimize `network`

    for layer in network.layers:
        result.values.add layer
        result.values[^1].bias.setZero
        result.values[^1].weights.setZero

    result.beta = beta
    result.lr = lr

func inputGradient*(backpropInfo: BackpropInfo): seq[Float] =
    ## Returns the input gradient that has been calculated during a backward pass.
    ## The parameter `calculateInputGradient` must be set to `true` for that,
    ## when calling `backward <#backward,Network,openArray[Float],BackpropInfo,staticbool>`_.

    doAssert backpropInfo.layers.len >= 1, "BackpropInfo needs at least one input layer and one output layer"
    doAssert backpropInfo.layers[0].inputGradient.len > 0, "Input gradient needs to be calculated before being used"
    backpropInfo.layers[0].inputGradient

func addGradient*(network: var Network, backpropInfo: BackpropInfo, lr: Float) =
    ## Applies the gradient accumulated during backwards passes in `backpropInfo` to `network`.
    ## Learning rate can be specified with `lr`.

    assert network.layers.len == backpropInfo.layers.len

    for layerIndex in 0..<network.layers.len:
        assert network.layers[layerIndex].bias.len == backpropInfo.layers[layerIndex].biasGradient.len
        assert network.layers[layerIndex].weights.len == backpropInfo.layers[layerIndex].weightsGradient.len
        
        network.layers[layerIndex].bias -= (lr / backpropInfo.numSummedGradients.Float) * backpropInfo.layers[layerIndex].biasGradient

        network.layers[layerIndex].weights -= (lr / backpropInfo.numSummedGradients.Float) * backpropInfo.layers[layerIndex].weightsGradient

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
            ((1 - moments.beta) / backpropInfo.numSummedGradients.Float) *
            backpropInfo.layers[layerIndex].biasGradient
        network.layers[layerIndex].bias -= moments.lr * moments.values[layerIndex].bias

        moments.values[layerIndex].weights *= moments.beta
        moments.values[layerIndex].weights += 
            ((1 - moments.beta) / backpropInfo.numSummedGradients.Float) *
            backpropInfo.layers[layerIndex].weightsGradient
        network.layers[layerIndex].weights -= moments.lr * moments.values[layerIndex].weights

func newLayer(numInputs, numOutputs: int, activation: ActivationFunction): Layer =
    assert numInputs > 0 and numOutputs > 0

    Layer(
        bias: newSeq[Float](numOutputs),
        weights: newSeq[Float](numInputs * numOutputs),
        numInputs: numInputs,
        numOutputs: numOutputs,
        activation: activation
    )

func initKaimingNormal*(network: var Network, randState: var Rand) =
    ## Randomly initializes the parameters of `network` using a method described in
    ## `"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    ## <https://arxiv.org/abs/1502.01852>`_.

    for layer in network.layers.mitems:
        layer.bias = newSeq[Float](layer.bias.len) # set bias to zero

        let std = sqrt(2.Float / layer.numInputs.Float)

        for v in layer.weights.mitems:
            v = randState.gauss(mu = 0.0, sigma = std)

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

template subNetwork*(network: Network, slice: untyped): Network =
    ## Creates new instance of `Network` from specified layers of an exising network.

    Network(layers: network.layers[slice])

func `%`(f: proc (x: Float): Float{.closure, noSideEffect.}): JsonNode =
  newJNull()
  
func initFromJson(dst: var ActivationFunction, jsonNode: JsonNode, jsonPath: var string) =
    let name = jsonNode{"name"}.getStr()
    dst = name.getActivationFunction()

func toJsonString*(network: Network): string =
    ## Returns the JSON represenation of `network`.

    {.cast(noSideEffect).}:
        let a = %*(network)
        pretty(a)

func toNetwork*(jsonString: string): Network =
    ## Creates a `Network` from a JSON formatted string representation.

    {.cast(noSideEffect).}:
        let jsonNode = jsonString.parseJson
    result = to(jsonNode, Network)

func `$`*(network: Network): string =
    ## Returns a string conveying some basic information about the structure  of the network.

    if network.layers.len == 0:
        return
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

    result &= "input:  " & getNumberString(network.layers[0].numInputs) & " neurons\n"

    for i, layer in network.layers.pairs:
        result &= "        "
        result &= spacesBeforeWeights & (
            if layer.numOutputs > layer.numInputs: "/\\\n"
            elif layer.numOutputs < layer.numInputs: "\\/\n"
            else: "||\n"
        )
        let isLast = i == network.layers.len - 1
        if isLast:
            result &= "output: "
        else:
            result &= "hidden: "

        result &= getNumberString(layer.numOutputs) & " neurons -> " & layer.activation.name
        
        if not isLast:
            result &= "\n"

func weightIndex*(inNeuron, outNeuron, numInputs, numOutputs: int): int =
    outNeuron + numOutputs * inNeuron

#----------- Feed Forward Functions -----------#

func feedForwardLayer(
    layer: Layer,
    input: openArray[Float],
    layerBackpropInfo: var (Nothing or LayerBackpropInfo)
): seq[Float] =

    assert input.len == layer.numInputs
    assert layer.bias.len == layer.numOutputs
    assert layer.weights.len == layer.numInputs * layer.numOutputs

    result = layer.bias

    # matrix-vector-multiplication
    # Y = alpha*A*X + beta*Y
    nimblas.gemv(
        order = colMajor,
        TransA = noTranspose,
        M = layer.numOutputs,
        N = layer.numInputs,
        alpha = 1.Float,
        A = addr layer.weights[0],
        lda = layer.numOutputs,
        X = addr input[0],
        incX = 1,
        beta = 1.Float,
        Y = addr result[0],
        incY = 1
    )

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.preActivation = result

    for value in result.mitems:
        value = layer.activation.f(value)

    when layerBackpropInfo isnot Nothing:
        layerBackpropInfo.postActivation = result

func feedForwardLayer(
    layer: Layer,
    input: openArray[SparseElement],
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

func forwardInternal(
    network: Network,
    input: openArray[Float] or openArray[SparseElement],
    backpropInfo: var (BackpropInfo or Nothing)
): seq[Float] =

    doAssert network.layers.len >= 1, "Network needs at least one input layer and one output layer"

    result = newSeq[Float](network.layers[^1].numOutputs)

    doAssert result.len == network.layers[^1].numOutputs, "Output size and output size of last layer must be the same"
    doAssert(
        input.len == network.layers[0].numInputs or input isnot openArray[Float],
        fmt"Input size ({input.len}) and input size of first layer ({network.layers[0].numInputs}) must be the same"
    )

    when backpropInfo isnot Nothing:
        doAssert backpropInfo.layers.len == network.layers.len
        when input is openArray[Float]:
            backpropInfo.input = input.toSeq
            backpropInfo.sparseInput.setLen(0)
        else:
            backpropInfo.sparseInput = input.toSeq
            backpropInfo.input.setLen(0)


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
    ## Runs the network with a given input. Also collects information in `backpropInfo` for a
    ## `backward <#backward,Network,openArray[Float],BackpropInfo,staticbool>`_ pass.
    ## Returns a `seq` with the length according to the size of the output layer of the network.
    ## 
    ## Note: When no information for a backward pass is needed,
    ## use `forward(Network, openArray[Float]) <#forward,Network,openArray[Float]>`_ for better performance.

    network.forwardInternal(input, backpropInfo)

func forward*(network: Network, input: openArray[Float]): seq[Float] =
    ## Runs the network with a given input. Does not collect information for a backward pass.
    ## Returns a `seq` with the length according to the size of the output layer of the network.

    var nothing: Nothing
    network.forwardInternal(input, nothing)

func forward*(network: Network, input: openArray[SparseElement], backpropInfo: var BackpropInfo): seq[Float] =
    ## Runs the network with a given sparse input. Also collects information in `backpropInfo` for a
    ## `backward <#backward,Network,openArray[Float],BackpropInfo,staticbool>`_ pass.
    ## Returns a `seq` with the length according to the size of the output layer of the network.
    ## 
    ## Note: When no information for a backward pass is needed,
    ## use `forward(Network, openArray[SparseElement]) <#forward,Network,openArray[SparseElement]>`_ for better performance.

    network.forwardInternal(input, backpropInfo)

func forward*(network: Network, input: openArray[SparseElement]): seq[Float] =
    ## Runs the network with a given sparse input. Does not collect information for a backward pass
    ## Returns a `seq` with the length according to the size of the output layer of the network.
    ## 
    ## A sparse input consist of a `seq` which contains tuples of indices and values.
    ## The indices indicate which input neuron is meant to have the according value.
    ## Sparse inputs are especially useful if inputs often have many zero elements. Then
    ## using sparse inputs can improve performance of the first layer significantly.

    var nothing: Nothing
    network.forwardInternal(input, nothing)

#----------- Back Propagate Functions -----------#

func backPropagateLayer(
    layer: Layer,
    outGradient: openArray[Float],
    inPostActivation: openArray[Float],
    layerBackpropInfo: var LayerBackpropInfo,
    calculateInputGradient: bool = true
) =
    assert layerBackpropInfo.biasGradient.len == layer.numOutputs
    assert layerBackpropInfo.weightsGradient.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert inPostActivation.len == layer.numInputs
    assert layerBackpropInfo.inputGradient.len == layer.numInputs

    for inNeuron in 0..<layer.numInputs:
        layerBackpropInfo.inputGradient[inNeuron] = 0

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.biasGradient[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]

    # outer product (vector-vector multiplication)
    # A = alpha*X*Y + A
    nimblas.ger(
        order = rowMajor,
        M = layer.numInputs,
        N = layer.numOutputs,
        alpha = 1.Float,
        X = addr inPostActivation[0],
        incX = 1,
        Y = addr layerBackpropInfo.biasGradient[0],
        incY = 1,
        A = addr layerBackpropInfo.weightsGradient[0],
        lda = layer.numOutputs
    )
    
    if calculateInputGradient:
        
        # Y = alpha*A*X + beta*Y
        nimblas.gemv(
            order = colMajor,
            TransA = transpose,
            M = layer.numOutputs,
            N = layer.numInputs,
            alpha = 1.Float,
            A = addr layer.weights[0],
            lda = layer.numOutputs,
            X = addr layerBackpropInfo.biasGradient[0],
            incX = 1,
            beta = 1.Float,
            Y = addr layerBackpropInfo.inputGradient[0],
            incY = 1
        )

func backPropagateLayer(
    layer: Layer,
    outGradient: openArray[Float],
    inPostActivation: openArray[SparseElement],
    layerBackpropInfo: var LayerBackpropInfo,
    calculateInputGradient: static bool = true
) =
    assert layerBackpropInfo.biasGradient.len == layer.numOutputs
    assert layerBackpropInfo.weightsGradient.len == layer.numInputs * layer.numOutputs
    assert outGradient.len == layer.numOutputs
    assert inPostActivation.len == layer.numInputs or inPostActivation is openArray[SparseElement]
    assert layerBackpropInfo.inputGradient.len == layer.numInputs

    for inNeuron in 0..<layer.numInputs:
        layerBackpropInfo.inputGradient[inNeuron] = 0

    for outNeuron in 0..<layer.numOutputs:
        layerBackpropInfo.biasGradient[outNeuron] =
            layer.activation.df(layerBackpropInfo.preActivation[outNeuron]) * outGradient[outNeuron]

    for (inNeuron, value) in inPostActivation:
        assert inNeuron in 0..<layer.numInputs
        for outNeuron in 0..<layer.numOutputs:
            let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
            layerBackpropInfo.weightsGradient[i] +=
                value * layerBackpropInfo.biasGradient[outNeuron]

    if calculateInputGradient:
        for inNeuron in 0..<layer.numInputs:
            for outNeuron in 0..<layer.numOutputs:
                let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
                layerBackpropInfo.inputGradient[inNeuron] +=
                    layer.weights[i] * layerBackpropInfo.biasGradient[outNeuron]

func backward*(
    network: Network,
    lossGradient: openArray[Float],
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

    if backpropInfo.input.len > 0:
        propagateLast(backpropInfo.input)
    else:
        doAssert backpropInfo.sparseInput.len > 0, "BackpropInfo is not set up correctly for backpropagation"
        propagateLast(backpropInfo.sparseInput)

    backpropInfo.numSummedGradients += 1