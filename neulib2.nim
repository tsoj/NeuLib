import std/[
    math,
    sequtils,
    random,
    json,
    strformat,
    streams
]


when (compiles do: import zippy/gzip):
  import zippy/gzip
  const useZippy = true
else:
  const useZippy = false

static: echo "Using zippy for file compression: ", useZippy

type
    ActivationFunction = enum
        relu, leakyRelu, sigmoid, tanh
    Layer*[T: SomeNumber] = object
        bias*: seq[T]
        weights*: seq[T]
        numInputs*: int
        numOutputs*: int
        activation*: ActivationFunction
    Network*[T: SomeNumber] = object
        layers*: seq[Layer[T]]
    SparseElement*[T: SomeNumber] = tuple[index: int, value: SomeNumber]
    LayerBackpropInfo*[T: SomeNumber] = object
        biasGradient*: seq[T]
        weightsGradient*: seq[T]
        preActivation: seq[T]
        postActivation: seq[T]
        inputGradient: seq[T]
    BackpropInfo*[T: SomeNumber] = object
        layers*: seq[LayerBackpropInfo[T]]
        numSummedGradients*: int
        case isSparseInput: bool
        of true: sparseInput: seq[SparseElement[T]]
        of false: input: seq[T]

#----------- Utility Functions -----------#

func toSparse*[T: SomeNumber](input: openArray[T], margin: T): seq[SparseElement] =
    ## Returns a sparse representation of a given array. Elements that are closer to zero than `margin` will not
    ## be added to the result.

    for i, a in input.pairs:
        if abs(a) >= margin:
            result.add((i, a))

func apply*[T: SomeNumber](x: var openArray[T], y: openArray[T], op: proc(x: var T, y: T) {.noSideEffect.}) =
    doAssert x.len == y.len
    for i in 0..<x.len:
        op(x[i], y[i])

func `+=`*[T: SomeNumber](x: var openArray[T], y: openArray[T]) =
    apply(x,y, proc(x: var T, y: T) = x += y)
func `-=`*[T: SomeNumber](x: var openArray[T], y: openArray[T]) =
    apply(x,y, proc(x: var T, y: T) = x -= y)
func `*=`*[T: SomeNumber](x: var openArray[T], y: openArray[T]) =
    apply(x,y, proc(x: var T, y: T) = x *= y)
func `/=`*[T: SomeNumber](x: var openArray[T], y: openArray[T]) =
    apply(x,y, proc(x: var T, y: T) = x /= y)

func `+=`*[T: SomeNumber](x: var openArray[T], y: T) =
    apply(x, proc(x: T): T = x + y)
func `-=`*[T: SomeNumber](x: var openArray[T], y: T) =
    apply(x, proc(x: T): T = x - y)
func `*=`*[T: SomeNumber](x: var openArray[T], y: T) =
    apply(x, proc(x: T): T = x * y)
func `/=`*[T: SomeNumber](x: var openArray[T], y: T) =
    apply(x, proc(x: T): T = x / y)

func `+`*[T: SomeNumber](x, y: openArray[T]): seq[T] =
    result &= x
    result += y
func `-`*[T: SomeNumber](x, y: openArray[T]): seq[T] =
    result &= x
    result -= y
func `*`*[T: SomeNumber](x, y: openArray[T]): seq[T] =
    result &= x
    result *= y
func `/`*[T: SomeNumber](x, y: openArray[T]): seq[T] =
    result &= x
    result /= y

func `+`*[T: SomeNumber](x: openArray[T], y: T): seq[T] =
    result &= x
    result += y
func `-`*[T: SomeNumber](x: openArray[T], y: T): seq[T] =
    result &= x
    result -= y
func `+`*[T: SomeNumber](x: T, y: openArray[T]): seq[T] =
    y + x
func `-`*[T: SomeNumber](x: T, y: openArray[T]): seq[T] =
    result &= y
    apply(result, proc(a: T): T = x - a)

func `*`*[T: SomeNumber](x: openArray[T], y: T): seq[T] =
    result &= x
    result *= y
func `/`*[T: SomeNumber](x: openArray[T], y: T): seq[T] =
    result &= x
    result /= y
func `*`*[T: SomeNumber](x: T, y: openArray[T]): seq[T] =
    y * x
func `/`*[T: SomeNumber](x: T, y: openArray[T]): seq[T] =
    result &= y
    apply(result, proc(a: T): T = x/a)

#----------- Activation and Loss Functions -----------#

func getActivationFunction*(activation: ActivationFunction, T: typedesc[SomeNumber]): auto =
    case activation:
    of relu: return proc(x: T): T = (if x > 0: x else: 0)
    of leakyRelu: return proc(x: T): T = (if x > 0: x else: 0.01.T * x)
    of sigmoid: 
        when T is SomeInteger:
            doAssert false, "simgoid not available as integer function"
        else:
            return proc(x: T): T = (1.T / (1.T + exp(-x)))
    of tanh: 
        when T is SomeInteger:
            doAssert false, "simgoid not available as integer function"
        else:
            return proc(x: T): T = (x.tanh)

func getActivationFunctionDerivative*(activation: ActivationFunction, T: typedesc[SomeNumber]): auto =
    static: doAssert T is SomeFloat, "Derivative not available as integer function"
    case activation:
    of relu: return proc(x: T): T = (if x > 0: 1 else: 0)
    of leakyRelu: return proc(x: T): T = (if x > 0: 1 else: 0.01)
    of sigmoid: return proc(x: T): T =
        let t = 1.T / (1.T + exp(-x))
        t * (1.T - t)
    of tanh: return proc(x: T): T = 1.T - pow(tanh(x), 2.T)

func customLossGradient*[TOut: SomeNumber, TIn](
    target: openArray[TIn],
    output: openArray[TOut],
    calculateElementLossGradient: proc(t: TIn, o: TOut): TOut {.noSideEffect.}
): seq[TOut] =
    ## Returns a custom loss gradient.
    ## `target` is the desired output vector, `output` is the actual output.

    result = newSeq[TOut](target.len)

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len
    assert output.len == result.len

    for i in 0..<target.len:
        result[i] = calculateElementLossGradient(target[i], output[i])

func customLoss*[TOut: SomeNumber, TIn](
    target: openArray[TIn],
    output: openArray[TOut],
    calculateElementLoss: proc(t: TIn, o: TOut): TOut {.noSideEffect.}
): TOut =
    ## Returns a custom loss value.
    ## `target` is the desired output vector, `output` is the actual output.

    assert target.len == output.len, "target.len: " & $target.len & ", output.len: " & $output.len

    for i in 0..<target.len:
        result += calculateElementLoss(target[i], output[i])
    result /= target.len.Float

func mseGradient*[T: SomeNumber](
    target: openArray[T],
    output: openArray[T]
): seq[T] =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLossGradient(target, output, proc(t, o: T): T = 2.T * (o - t))

func mseLoss*[T: SomeNumber](
    target: openArray[T],
    output: openArray[T]
): T =
    ## Returns the gradient of the mean squared error loss function.
    ## `target` is the desired output vector, `output` is the actual output.
 
    customLoss(target, output, proc(t, o: T): T = (o - t)*(o - t))

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

func newBackpropInfo*[T: SomeNumber](network: Network[T]): BackpropInfo[T] =
    ## Creates `BackpropInfo` that can be used to do backwards passes with `network`.

    for layer in network.layers:
        result.layers.add(LayerBackpropInfo(
            biasGradient: layer.bias,
            weightsGradient: layer.weights,
            inputGradient: newSeq[T](layer.numInputs)
        ))
    result.setZero

func inputGradient*[T: SomeNumber](backpropInfo: BackpropInfo[T]): seq[T] =
    ## Returns the input gradient that has been calculated during a backward pass.
    ## The parameter `calculateInputGradient` must be set to `true` for that,
    ## when calling `backward <#backward,Network,openArray[Float],BackpropInfo,staticbool>`_.

    doAssert backpropInfo.layers.len >= 1, "BackpropInfo needs at least one input layer and one output layer"
    doAssert backpropInfo.layers[0].inputGradient.len > 0, "Input gradient needs to be calculated before being used"
    backpropInfo.layers[0].inputGradient

func addGradient*[T: SomeNumber](network: var Network[T], backpropInfo: BackpropInfo[T], lr: T) =
    ## Applies the gradient accumulated during backwards passes in `backpropInfo` to `network`.
    ## Learning rate can be specified with `lr`.

    assert network.layers.len == backpropInfo.layers.len

    for layerIndex in 0..<network.layers.len:
        assert network.layers[layerIndex].bias.len == backpropInfo.layers[layerIndex].biasGradient.len
        assert network.layers[layerIndex].weights.len == backpropInfo.layers[layerIndex].weightsGradient.len
        
        network.layers[layerIndex].bias -= (lr / backpropInfo.numSummedGradients.T) * backpropInfo.layers[layerIndex].biasGradient

        network.layers[layerIndex].weights -= (lr / backpropInfo.numSummedGradients.T) * backpropInfo.layers[layerIndex].weightsGradient

func newLayer(numInputs, numOutputs: int, activation: ActivationFunction, T: typedesc[SomeNumber]): Layer[T] =
    assert numInputs > 0 and numOutputs > 0

    Layer[T](
        bias: newSeq[T](numOutputs),
        weights: newSeq[T](numInputs * numOutputs),
        numInputs: numInputs,
        numOutputs: numOutputs,
        activation: activation
    )

func initKaimingNormal*[T: SomeNumber](network: var Network[T], randState: var Rand) =
    ## Randomly initializes the parameters of `network` using a method described in
    ## `"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    ## <https://arxiv.org/abs/1502.01852>`_.

    for layer in network.layers.mitems:
        layer.bias = newSeq[T](layer.bias.len) # set bias to zero

        let std = sqrt(2.T / layer.numInputs.T)

        for v in layer.weights.mitems:
            v = randState.gauss(mu = 0.0, sigma = std.float).T

func initKaimingNormal*(network: var Network, seed: int64 = 0) =
    ## Randomly initializes the parameters of `network` using a method described in
    ## `"Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    ## <https://arxiv.org/abs/1502.01852>`_.

    var randState: Rand = initRand(seed)
    network.initKaimingNormal(randState)

func newNetwork*[T: SomeNumber](
    input: int,
    ls: varargs[tuple[numNeurons: int, activation: ActivationFunction]]
): Network[T] =
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

    doAssert ls.len >= 1, "Network needs at least one layer"

    for i in 0..<ls.len:
        result.layers.add(newLayer(
            numInputs = if i == 0: input else: ls[i-1].numNeurons,
            numOutputs = ls[i].numNeurons,
            activation = ls[i].activation,
            T = T
        ))

    result.initKaimingNormal(seed = 0)

func description*[T: SomeNumber](network: Network[T]): string =
    ## Returns a string conveying some basic information about the structure  of the network.

    if network.layers.len == 0:
        return

    result &= "Number type: " & $T & "\n"

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

func toNetwork*(jsonString: string, T: typedesc[SomeNumber]): Network[T] =
    ## Creates a `Network` from a JSON formatted string representation.

    {.cast(noSideEffect).}:
        let jsonNode = jsonString.parseJson
    result = jsonNode.to Network[T]

proc writeSeq[T](stream: Stream, x: seq[T], writeProc: proc(s: Stream, x: T)) =
    stream.write x.len.int64
    for a in x:
        stream.writeProc a
    
proc readSeq[T](stream: Stream, r: var seq[T], readProc: proc(s: Stream, x: var T)) =
    let len = stream.readInt64
    for i in 0..<len:
        var element: T
        stream.readProc element
        r.add element

proc writeLayer[T: SomeNumber](stream: Stream, layer: Layer[T]) =
    stream.writeSeq layer.bias, write
    stream.writeSeq layer.weights, write
    stream.write layer.numInputs.int64
    stream.write layer.numOutputs.int64
    stream.write layer.activation.int64
    
proc readLayer[T: SomeNumber](stream: Stream, layer: var Layer[T]) =
    stream.readSeq layer.bias, read
    stream.readSeq layer.weights, read
    layer.numInputs = stream.readInt64
    layer.numOutputs = stream.readInt64
    layer.activation = stream.readInt64.ActivationFunction

proc writeNetwork*[T: SomeNumber](stream: Stream, network: Network[T]) =
    stream.writeLine($T)
    stream.writeSeq network.layers, writeLayer

proc readNetwork*[T: SomeNumber](stream: Stream, network: var Network[T]) =
    let streamNetworkBaseType = stream.readLine
    doAssert streamNetworkBaseType == $T, "Stream is a representation of Network[" & streamNetworkBaseType & "], but expected a Network[" & $T & "]"

    stream.readSeq network.layers, readLayer

when useZippy:
    proc writeNetworkCompressed*[T: SomeNumber](stream: FileStream, network: Network[T]) =
        discard

    proc readNetworkCompressed*[T: SomeNumber](stream: FileStream, network: var Network[T]) =
        discard

proc saveToFile*[T: SomeNumber](network: Network[T], fileName: string) =
    var fileStream = newFileStream(fileName, fmWrite)
    if fileStream.isNil:
        raise newException(IOError, "Couldn't open file: " & fileName)

    when useZippy:
        fileStream.writeNetworkCompressed network
    else:
        fileStream.writeNetwork network
    fileStream.close

proc loadFromFile*[T](network: var Network[T], fileName: string) =
    var fileStream = newFileStream(fileName, fmRead)
    if fileStream.isNil:
        raise newException(IOError, "Couldn't open file: " & fileName)

    when useZippy:
        fileStream.readNetworkCompressed network
    else:
        fileStream.readNetwork network
    fileStream.close

var network = newNetwork[float32](100, (32, relu), (1, sigmoid))
echo network.description
writeFile "test.json", network.toJsonString

network.saveToFile "test2.bin"

var network2: Network[float32]
network2.loadFromFile "test4.bin"
echo network2.description
writeFile "test2.json", network2.toJsonString
