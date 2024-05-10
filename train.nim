import neulib, mnist

import std/[sequtils, random, times, strformat, math]

const
  startLr = 0.05
  finalLr = 0.01
  numEpochs = 10

let
  trainX = trainImages.toSeqs(float32)
  testX = testImages.toSeqs(float32)
  trainY = trainLabels.toSeqs(float32)
  testY = testLabels.toSeqs(float32)

let lrDecay = pow(finalLr / startLr, 1.0 / float(numEpochs * trainX.len))
doAssert startLr > finalLr,
  "Starting learning rate must be strictly bigger than the final learning rate"
doAssert finalLr == startLr or lrDecay < 1.0,
  "lrDecay should be smaller than one if the learning rate should decrease"

type WhichInputTypeToUse = enum
  useSparseInputs
  useSparseOneInputs
  useFullInputs

for whatInputType in (useSparseInputs, useSparseOneInputs, useFullInputs).fields:
  echo "Beginning with training ..."
  echo "Inputs: ", whatInputType

  var model = newNetwork(28 * 28, (64, relu), (32, relu), (10, sigmoid))

  # Or load from file
  # var model = readFile("mnist_model.json").toNetwork
  # var model = loadNetworkFromFile "mnist_model.bin"

  echo model.description

  var
    lr = startLr
    backpropInfo: BackpropInfo

  for epoch in 1 .. numEpochs:
    let start = now()

    var shuffledIndices = (0 ..< trainX.len).toSeq
    shuffledIndices.shuffle

    for index in shuffledIndices:
      let
        input =
          when whatInputType == useSparseInputs:
            trainX[index].toSparse
          elif whatInputType == useSparseOneInputs:
            trainX[index].toSparseOnes(splitAt = 0.5)
          else:
            trainX[index]

        output = model.forward(input, backpropInfo)

      let lossGradient = mseGradient(target = trainY[index], output = output)

      model.backward(lossGradient, backpropInfo, lr)
      lr *= lrDecay

    echo fmt"Finished epoch {epoch} in {(now() - start).inMilliseconds} ms, lr: {lr:.6f}"

    var numCorrect = 0
    doAssert testX.len == testY.len
    for i in 0 ..< testX.len:
      let input =
        when whatInputType == useSparseInputs:
          testX[i].toSparse
        elif whatInputType == useSparseOneInputs:
          testX[i].toSparseOnes(splitAt = 0.5)
        else:
          testX[i]
      let output = model.forward(input)
      if output.maxIndex == testY[i].maxIndex:
        numCorrect += 1
    echo "Neural net decided ",
      fmt"{100.0*numCorrect.float/testX.len.float:.2f}", " % test cases correctly."

  # Might want to store trained model
  # Store model as JSON
  # writeFile "mnist_model.json", model.toJsonString
  # Or store trained model as binary
  # model.saveToFile "mnist_model.bin"
