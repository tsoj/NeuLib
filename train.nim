import neulib, mnist

import std/[sequtils, random, times, strformat]

const
    batchSize = 15
    lr = 0.2

let
    trainX = trainImages.toSeqs(Float)
    testX = testImages.toSeqs(Float)
    trainY = trainLabels.toSeqs(Float)
    testY = testLabels.toSeqs(Float)

const useMoments = false

for usingSparseInputs in (false, true).fields:


    echo "Beginning with training ..."
    echo "Using ", if usingSparseInputs: "sparse" else: "full", " inputs"

    var model = newNetwork(
        28*28,
        (64, relu),
        (32, relu),
        (10, sigmoid)
    )

    # or load from file
    # var model = readFile("mnist_model.json").toNetwork

    echo model

    var
        backpropInfo = model.newBackpropInfo
        moments = model.newMoments(lr = lr, beta = 0.9)



    for epoch in 0..<10:
        let start = now()

        var shuffledIndices = (0..<trainX.len).toSeq
        shuffledIndices.shuffle

        for batch in 0..<(shuffledIndices.len div batchSize):
            
            backpropInfo.setZero

            for i in 0..<batchSize:
                let index = shuffledIndices[batch * batchSize + i]

                let  output = model.forward(
                    when usingSparseInputs: trainX[index].toSparse else: trainX[index], 
                    backpropInfo
                )

                let lossGradient = mseGradient(target = trainY[index], output = output)

                model.backward(lossGradient, backpropInfo)
            
            if useMoments:
                model.addGradient(backpropInfo, moments)
            else:
                model.addGradient(backpropInfo, lr)

        echo(
            "Finished epoch ", epoch, " in ", (now() - start).inMilliseconds, " ms (used ",
            if usingSparseInputs: "sparse" else: "full", " inputs)"
        )


        var numCorrect = 0
        doAssert testX.len == testY.len
        for i in 0..<testX.len:
            let output = model.forward(when usingSparseInputs: testX[i].toSparse else: testX[i])
            if output.maxIndex == testY[i].maxIndex:
                numCorrect += 1
        echo "Neural net decided ", fmt"{100.0*numCorrect.float/testX.len.float:.2f}", " % test cases correctly."

    # might want to store trained model
    # writeFile("mnist_model.json", model.toJsonString)