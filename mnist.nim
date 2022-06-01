import streams, endians, unicode, sequtils, random, os
import framebuffer

proc readInt32BE(stream: Stream): int32 {.inline.}=
  var rawBytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

type Image = array[28, array[28, uint8]]

proc loadImages(filename: string): seq[Image] =

    var stream: Stream = newFileStream(filename, mode = fmRead)
    defer: stream.close

    let magicNumber = stream.readInt32BE
    doAssert magicNumber == 2051'i32, "This file is not a MNIST images file."

    let
        numImgs = stream.readInt32BE.int
        numRows = stream.readInt32BE.int
        numCols = stream.readInt32BE.int

    doAssert numRows == 28 and numCols == 28
    result.setLen(numImgs)
    discard stream.readData(addr result[0][0][0], result.len * sizeof result[0])

proc loadLabels(filename: string): seq[uint8] =

    var stream: Stream = newFileStream(filename, mode = fmRead)
    defer: stream.close

    let
        magicNumber = stream.readInt32BE
        numLabels = stream.readInt32BE.int
    doAssert magicNumber == 2049'i32, "This file is not a MNIST labels file."

    result = newSeq[uint8](numLabels)
    discard stream.readData(addr result[0], result.len * sizeof result[0])

func imageToRuneBox(image: Image, label: uint8): seq[seq[Rune]] =
    result = newSeq[seq[Rune]](29)
    for h in 0..<28:
        result[h] = newSeq[Rune](28)
        for w in 0..<28:
            result[h][w] = " ░▒▓█".toRunes[image[h][w] div (uint8.high div 4)]
    result[^1] = ($label).toRunes

func toSeq(image: Image, T: typedesc): seq[T] =
    for h in 0..<28:
        for w in 0..<28:
            result.add(image[h][w].T / uint8.high.T)

func toSeqs*(images: seq[Image], T: typedesc): seq[seq[T]] =
    for image in images:
        result.add image.toSeq(T)

func toSeqs*(labels: seq[uint8], T: typedesc): seq[seq[T]] =
    for label in labels:
        result.add newSeq[T](10)
        doAssert label.int in 0..9
        result[^1][label.int] = 1.T

let
    trainImages* = loadImages("./mnist_data/train-images-idx3-ubyte")
    testImages* = loadImages("./mnist_data/t10k-images-idx3-ubyte")
    trainLabels* = loadLabels("./mnist_data/train-labels-idx1-ubyte")
    testLabels* = loadLabels("./mnist_data/t10k-labels-idx1-ubyte")

doAssert trainImages.len == trainLabels.len
doAssert testImages.len == testLabels.len        

when isMainModule:

    var fb = newFramebuffer()

    let
        images = testImages
        labels = testLabels

    var shuffledIndices = (0..<images.len).toSeq
    shuffledIndices.shuffle
    for i in shuffledIndices:
        fb.add(
            imageToRuneBox(images[i], labels[i]),
            0,0
        )

        fb.print()

        sleep(1000)