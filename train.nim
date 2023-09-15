import neulib, mnist

import std/[sequtils, random, times, strformat]

const
    batchSize = 15
    lr = 0.2
    useMoments = false

let
    trainX = trainImages.toSeqs(float32)
    testX = testImages.toSeqs(float32)
    trainY = trainLabels.toSeqs(float32)
    testY = testLabels.toSeqs(float32)

type WhichInputTypeToUse = enum
    useSparseInputs, useSparseOneInputs, useFullInputs
for whatInputType in (useSparseInputs, useSparseOneInputs, useFullInputs).fields:

    echo "Beginning with training ..."
    echo "Inputs: ", whatInputType

    var model = newNetwork(
        28*28,
        (64, relu),
        (32, relu),
        (10, sigmoid)
    )

    # Or load from file
    # var model = readFile("mnist_model.json").toNetwork
    # var model = loadNetworkFromFile "mnist_model.bin"

    echo model.description

    var
        backpropInfo = model.newBackpropInfo
        moments = model.newMoments(lr = lr, beta = 0.5)

    for epoch in 0..<10:
        let start = now()

        var shuffledIndices = (0..<trainX.len).toSeq
        shuffledIndices.shuffle

        for batch in 0..<(shuffledIndices.len div batchSize):
            
            backpropInfo.setZero

            for i in 0..<batchSize:
                let index = shuffledIndices[batch * batchSize + i]

                let
                    input = when whatInputType == useSparseInputs:
                        trainX[index].toSparse
                    elif whatInputType == useSparseOneInputs:
                        trainX[index].toSparseOnes(splitAt = 0.5)
                    else:
                        trainX[index]
                        
                    output = model.forward(input, backpropInfo)

                let lossGradient = mseGradient(target = trainY[index], output = output)

                model.backward(lossGradient, backpropInfo)
            
            if useMoments:
                model.addGradient(backpropInfo, moments)
            else:
                model.addGradient(backpropInfo, lr)

        echo(
            "Finished epoch ", epoch, " in ", (now() - start).inMilliseconds, " ms (", whatInputType, ")"
        )


        var numCorrect = 0
        doAssert testX.len == testY.len
        for i in 0..<testX.len:
            let input = when whatInputType == useSparseInputs:
                testX[i].toSparse
            elif whatInputType == useSparseOneInputs:
                testX[i].toSparseOnes(splitAt = 0.5)
            else:
                testX[i]
            let output = model.forward(input)
            if output.maxIndex == testY[i].maxIndex:
                numCorrect += 1
        echo "Neural net decided ", fmt"{100.0*numCorrect.float/testX.len.float:.2f}", " % test cases correctly."

    # Might want to store trained model
    # Store model as JSON
    # writeFile "mnist_model.json", model.toJsonString
    # Or store trained model as binary
    # model.saveToFile "mnist_model.bin"