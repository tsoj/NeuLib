import neulib, mnist

import sequtils, random, times

var model = newNetwork(28*28, (40, relu), (40, relu), (10, sigmoid))

let batchSize = 15

let
    trainX = trainImages.toSeqs(Float)
    testX = testImages.toSeqs(Float)
    trainY = trainLabels.toSeqs(Float)
    testY = testLabels.toSeqs(Float)


var backpropInfo = model.getBackpropInfo

echo "Beginning with training ..."

for epoch in 0..<10:
    let start = now()

    var shuffledIndices = (0..<trainX.len).toSeq
    shuffledIndices.shuffle

    for batch in 0..<(shuffledIndices.len div batchSize):

        
        backpropInfo.setZero

        for i in 0..<batchSize:
            let index = shuffledIndices[batch * batchSize + i]

            let 
                output = model.forward(trainX[index], backpropInfo)
                lossGradient = mseGradient(target = trainY[index], output = output)

            model.backward(lossGradient, backpropInfo)
        model.addGradient(backpropInfo, -0.2)

    echo "Finished epoch ", epoch, " in ", now() - start