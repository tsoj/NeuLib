import std/[
    math,
    sequtils,
    random,
    json,
    strformat
]

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
        layers*: seq[T]
    SparseElement*[T: SomeNumber] = tuple[index: int, value: SomeNumber]
    LayerBackpropInfo*[T: SomeNumber] = object
        biasGradient*: seq[T]
        weightsGradient*: seq[T]
        preActivation: seq[T]
        postActivation: seq[T]
        inputGradient: seq[T]
    BackpropInfo*[T: SomeNumber] = object
        layers*: seq[LayerBackpropInfo[T]]
        input: seq[T]
        sparseInput: seq[SparseElement[T]]
        numSummedGradients*: int

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