import std/[math, sequtils, random, json, strformat, streams]

type
  ActivationFunction* = enum
    identity
    relu
    leakyRelu
    sigmoid
    tanh

  Layer* = object
    bias*: seq[float32]
    weights*: seq[float32]
    numInputs*: int
    numOutputs*: int
    activation*: ActivationFunction

  Network* = object
    layers*: seq[Layer]

  LayerBackpropInfo = object
    preActivation: seq[float32]
    postActivation: seq[float32]
    inputGradient: seq[float32]

  SparseElement* = tuple[index: int, value: float32]
  SparseOneElement* = int
  InputType = enum
    full
    sparse
    sparseOnes

  NetworkInput = object
    case isSparseInput: InputType
    of sparse: sparseInput: seq[SparseElement]
    of sparseOnes: sparseOnesInput: seq[SparseOneElement]
    of full: input: seq[float32]

  BackpropInfo* = object
    layers: seq[LayerBackpropInfo]
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
      result.add(i)

#----------- Activation and Loss Functions -----------#

func getActivationFunction*(activation: ActivationFunction): auto =
  case activation
  of identity:
    return proc(x: float32): float32 =
      x
  of relu:
    return proc(x: float32): float32 =
      (if x > 0: x else: 0)
  of leakyRelu:
    return proc(x: float32): float32 =
      (if x > 0: x else: 0.01f * x)
  of sigmoid:
    return proc(x: float32): float32 =
      (1.0f / (1.0f + exp(-x)))
  of tanh:
    return proc(x: float32): float32 =
      (x.tanh)

func getActivationFunctionDerivative*(activation: ActivationFunction): auto =
  case activation
  of identity:
    return proc(x: float32): float32 =
      1.0f
  of relu:
    return proc(x: float32): float32 =
      (if x > 0: 1 else: 0)
  of leakyRelu:
    return proc(x: float32): float32 =
      (if x > 0: 1 else: 0.01f)
  of sigmoid:
    return proc(x: float32): float32 =
      let t = 1.0f / (1.0f + exp(-x))
      t * (1.0f - t)
  of tanh:
    return proc(x: float32): float32 =
      1.0f - pow(x.tanh, 2.0f)

func customLossGradient*[TIn](
    target: openArray[TIn],
    output: openArray[float32],
    calculateElementLossGradient: proc(t: TIn, o: float32): float32 {.noSideEffect.},
): seq[float32] =
  ## Returns a custom loss gradient.
  ## `target` is the desired output vector, `output` is the actual output.

  result = newSeq[float32](target.len)

  assert target.len == output.len,
    "target.len: " & $target.len & ", output.len: " & $output.len
  assert output.len == result.len

  for i in 0 ..< target.len:
    result[i] = calculateElementLossGradient(target[i], output[i])

func customLoss*[TIn](
    target: openArray[TIn],
    output: openArray[float32],
    calculateElementLoss: proc(t: TIn, o: float32): float32 {.noSideEffect.},
): float32 =
  ## Returns a custom loss value.
  ## `target` is the desired output vector, `output` is the actual output.

  assert target.len == output.len,
    "target.len: " & $target.len & ", output.len: " & $output.len

  for i in 0 ..< target.len:
    result += calculateElementLoss(target[i], output[i])
  result /= target.len.float32

func mseGradient*(
    target: openArray[float32], output: openArray[float32]
): seq[float32] =
  ## Returns the gradient of the mean squared error loss function.
  ## `target` is the desired output vector, `output` is the actual output.

  customLossGradient(
    target,
    output,
    proc(t, o: float32): float32 =
      2.0f * (o - t)
    ,
  )

func mseLoss*(target: openArray[float32], output: openArray[float32]): float32 =
  ## Returns the gradient of the mean squared error loss function.
  ## `target` is the desired output vector, `output` is the actual output.

  customLoss(
    target,
    output,
    proc(t, o: float32): float32 =
      (o - t) * (o - t)
    ,
  )

#----------- Network and Gradient Stuff  -----------#

func inputGradient*(backpropInfo: BackpropInfo): seq[float32] =
  ## Returns the input gradient that has been calculated during a backward pass (
  ## when calling `backward <#backward,Network,openArray[float32],BackpropInfo,float32,staticbool>`_,
  ## the parameter `calculateInputGradient` must be set to `true`)

  doAssert backpropInfo.layers.len >= 1,
    "BackpropInfo needs at least one input layer and one output layer"
  backpropInfo.layers[0].inputGradient

func newLayer(numInputs, numOutputs: int, activation: ActivationFunction): Layer =
  assert numInputs > 0 and numOutputs > 0

  Layer(
    bias: newSeq[float32](numOutputs),
    weights: newSeq[float32](numInputs * numOutputs),
    numInputs: numInputs,
    numOutputs: numOutputs,
    activation: activation,
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
    input: int, layers: varargs[tuple[numNeurons: int, activation: ActivationFunction]]
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

  for i in 0 ..< layers.len:
    result.layers.add(
      newLayer(
        numInputs =
          if i == 0:
            input
          else:
            layers[i - 1].numNeurons
        ,
        numOutputs = layers[i].numNeurons,
        activation = layers[i].activation,
      )
    )

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
    for i in 0 ..< (maxLength + "neurons".len) div 2:
      s &= " "
    s

  result &= "Input:  " & getNumberString(network.layers[0].numInputs) & " neurons\n"

  for i, layer in network.layers.pairs:
    result &= "        "
    result &=
      spacesBeforeWeights & (
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

proc writeSeq[float32](
    stream: Stream, x: seq[float32], writeProc: proc(s: Stream, x: float32)
) =
  stream.write x.len.int64
  for a in x:
    stream.writeProc a

proc readSeq[float32](
    stream: Stream, r: var seq[float32], readProc: proc(s: Stream, x: var float32)
) =
  let len = stream.readInt64
  for i in 0 ..< len:
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

  try:
    stream.writeSeq network.layers, writeLayer
  except CatchableError:
    getCurrentException().msg = "Couldn't write network: " & getCurrentException().msg
    raise

proc readNetwork*(stream: Stream): Network =
  ## Loads a network from `stream`, assuming a binary representation as created
  ## by `writeNetwork <#writeNetwork,Stream,Network>`_.

  try:
    stream.readSeq result.layers, readLayer
  except CatchableError:
    getCurrentException().msg = "Couldn't read network: " & getCurrentException().msg
    raise

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
    layerBackpropInfo: var (Nothing or LayerBackpropInfo),
): seq[float32] =
  assert layer.bias.len == layer.numOutputs
  assert layer.weights.len == layer.numInputs * layer.numOutputs

  result = layer.bias

  when input is openArray[SparseElement]:
    for (inNeuron, value) in input:
      assert inNeuron in 0 ..< layer.numInputs
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        result[outNeuron] += layer.weights[i] * value
  elif input is openArray[SparseOneElement]:
    for inNeuron in input:
      assert inNeuron in 0 ..< layer.numInputs
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        result[outNeuron] += layer.weights[i]
  else:
    assert input.len == layer.numInputs
    for inNeuron in 0 ..< layer.numInputs:
      for outNeuron in 0 ..< layer.numOutputs:
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
    backpropInfo: var (BackpropInfo or Nothing),
): seq[float32] =
  doAssert network.layers.len >= 1,
    "Network needs at least one input layer and one output layer"

  result = newSeq[float32](network.layers[^1].numOutputs)

  doAssert result.len == network.layers[^1].numOutputs,
    "Output size and output size of last layer must be the same"
  doAssert(
    input.len == network.layers[0].numInputs or
      input is openArray[SparseElement or SparseOneElement],
    fmt"Input size ({input.len}) and input size of first layer ({network.layers[0].numInputs}) must be the same",
  )

  when backpropInfo isnot Nothing:
    backpropInfo.layers.setLen network.layers.len
    when input is openArray[SparseElement]:
      backpropInfo.input = NetworkInput(isSparseInput: sparse, sparseInput: input.toSeq)
    elif input is openArray[SparseOneElement]:
      backpropInfo.input =
        NetworkInput(isSparseInput: sparseOnes, sparseOnesInput: input.toSeq)
    else:
      static:
        doAssert input is openArray[float32]
      backpropInfo.input = NetworkInput(isSparseInput: full, input: input.toSeq)

  result = feedForwardLayer(
    layer = network.layers[0],
    input = input,
    layerBackpropInfo =
      when backpropInfo is Nothing:
        backpropInfo
      else:
        backpropInfo.layers[0]
    ,
  )

  for i in 1 ..< network.layers.len:
    result = feedForwardLayer(
      layer = network.layers[i],
      input = result,
      layerBackpropInfo =
        when backpropInfo is Nothing:
          backpropInfo
        else:
          backpropInfo.layers[i]
      ,
    )

func forward*(
    network: Network,
    input: openArray[float32 or SparseElement or SparseOneElement],
    backpropInfo: var BackpropInfo,
): seq[float32] =
  ## Runs the network with a given input. Also collects information in `backpropInfo` for a
  ## `backward <#backward,Network,openArray[float32],BackpropInfo,float32,staticbool>`_ pass.
  ## Returns a `seq` with the length according to the size of the output layer of the network.
  ## 
  ## Note: When no information for a backward pass is needed,
  ## use `forward(Network, openArray[]) <#forward,Network,openArray[]>`_ for better performance.

  network.forwardInternal(input, backpropInfo)

func forward*(
    network: Network, input: openArray[float32 or SparseElement or SparseOneElement]
): seq[float32] =
  ## Runs the network with a given input. Does not collect information for a backward pass.
  ## Returns a `seq` with the length according to the size of the output layer of the network.

  var nothing: Nothing
  network.forwardInternal(input, nothing)

func backPropagateLayer(
    layer: var Layer,
    lr: float32,
    outGradient: openArray[float32],
    inPostActivation: openArray[float32 or SparseElement or SparseOneElement],
    layerBackpropInfo: var LayerBackpropInfo,
    calculateInputGradient: static bool = true,
) =
  assert outGradient.len == layer.numOutputs

  var biasGradient = newSeq[float32](layer.numOutputs)
  for outNeuron in 0 ..< layer.numOutputs:
    biasGradient[outNeuron] =
      getActivationFunctionDerivative(layer.activation)(
        layerBackpropInfo.preActivation[outNeuron]
      ) * outGradient[outNeuron]

    layer.bias[outNeuron] -= biasGradient[outNeuron] * lr

  when calculateInputGradient:
    layerBackpropInfo.inputGradient = newSeq[float32](layer.numInputs)
    for inNeuron in 0 ..< layer.numInputs:
      layerBackpropInfo.inputGradient[inNeuron] = 0
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        layerBackpropInfo.inputGradient[inNeuron] +=
          layer.weights[i] * biasGradient[outNeuron]

  when inPostActivation is openArray[SparseElement]:
    for (inNeuron, value) in inPostActivation:
      assert inNeuron in 0 ..< layer.numInputs
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        layer.weights[i] -= value * biasGradient[outNeuron] * lr
  elif inPostActivation is openArray[SparseOneElement]:
    for inNeuron in inPostActivation:
      assert inNeuron in 0 ..< layer.numInputs
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        layer.weights[i] -= biasGradient[outNeuron] * lr
  else:
    assert inPostActivation.len == layer.numInputs
    for inNeuron in 0 ..< layer.numInputs:
      for outNeuron in 0 ..< layer.numOutputs:
        let i = weightIndex(inNeuron, outNeuron, layer.numInputs, layer.numOutputs)
        layer.weights[i] -= inPostActivation[inNeuron] * biasGradient[outNeuron] * lr

func backward*(
    network: var Network,
    lossGradient: openArray[float32],
    backpropInfo: var BackpropInfo,
    lr: float32,
    calculateInputGradient: static bool = false,
) =
  ## Calculates and adds the gradient of all network parameters in regard to the loss gradient to `backpropInfo`.
  ## When `calculateInputGradient` is set to false, the gradient for the input will not be calculated.

  doAssert network.layers.len >= 1,
    "Network needs at least one input layer and one output layer"
  doAssert lossGradient.len == network.layers[^1].numOutputs,
    "Loss size and output size of last layer must be the same"
  doAssert backpropInfo.layers.len == network.layers.len,
    "BackpropInfo is not set up correctly for backpropagation"

  for i in countdown(network.layers.len - 1, 1):
    backPropagateLayer(
      layer = network.layers[i],
      lr = lr,
      outGradient =
        if i == network.layers.len - 1:
          lossGradient
        else:
          backpropInfo.layers[i + 1].inputGradient
      ,
      inPostActivation = backpropInfo.layers[i - 1].postActivation,
      layerBackpropInfo = backpropInfo.layers[i],
    )

  template propagateLast(input: auto) =
    backPropagateLayer(
      layer = network.layers[0],
      lr = lr,
      outGradient =
        if backpropInfo.layers.len <= 1:
          lossGradient
        else:
          backpropInfo.layers[1].inputGradient
      ,
      inPostActivation = input,
      layerBackpropInfo = backpropInfo.layers[0],
      calculateInputGradient = calculateInputGradient,
    )

  case backpropInfo.input.isSparseInput
  of sparse:
    propagateLast(backpropInfo.input.sparseInput)
  of sparseOnes:
    propagateLast(backpropInfo.input.sparseOnesInput)
  of full:
    propagateLast(backpropInfo.input.input)
