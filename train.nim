import neulib, mnist

import sequtils, random, times, strformat

var model = newNetwork(
    28*28,
    (40, relu),
    (40, relu),
    (10, sigmoid)
)

# or load from file
# var model = readFile("mnist_model.json").toNetwork

echo model

let
    batchSize = 15
    lr = 0.2

let
    trainX = trainImages.toSeqs(Float)
    testX = testImages.toSeqs(Float)
    trainY = trainLabels.toSeqs(Float)
    testY = testLabels.toSeqs(Float)


var backpropInfo = model.newBackpropInfo

echo "Beginning with training ..."

for epoch in 0..<10:
    let start = now()

    var shuffledIndices = (0..<trainX.len).toSeq
    shuffledIndices.shuffle

    for batch in 0..<(shuffledIndices.len div batchSize):
        
        backpropInfo.setZero

        for i in 0..<batchSize:
            let index = shuffledIndices[batch * batchSize + i]

            var output: seq[Float]
            if rand(1.0) > 0.5:
                # normal input layout
                output = model.forward(trainX[index], backpropInfo)
            else:
                # try out sparse input layout
                output = model.forward(trainX[index].toSparse, backpropInfo)

            let lossGradient = mseGradient(target = trainY[index], output = output)

            model.backward(lossGradient, backpropInfo)
        model.addGradient(backpropInfo, lr)

    echo "Finished epoch ", epoch, " in ", (now() - start).inMilliseconds, " ms"


    var numCorrect = 0
    doAssert testX.len == testY.len
    for i in 0..<testX.len:
        let output = model.forward(testX[i].toSparse)
        if output.maxIndex == testY[i].maxIndex:
            numCorrect += 1
    echo "Neural net decided ", fmt"{100.0*numCorrect.float/testX.len.float:.2f}", " % test cases correctly."

# might want to store trained model
# writeFile("mnist_model.json", model.toJsonString)