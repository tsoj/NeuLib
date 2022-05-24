import streams, endians, unicode
import framebuffer

proc readInt32BE(stream: Stream): int32 {.inline.}=
  var rawBytes = stream.readInt32
  bigEndian32(addr result, addr raw_bytes)

proc loadImages(filename: string): seq[array[28, array[28, uint8]]] =

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
    echo magicNumber

    result = newSeq[uint8](numLabels)
    discard stream.readData(addr result[0], result.len * sizeof result[0])

let
    trainImages = loadImages("./mnist_data/train-images-idx3-ubyte")
    testImages = loadImages("./mnist_data/t10k-images-idx3-ubyte")
    trainLabels = loadLabels("./mnist_data/train-labels-idx1-ubyte")
    testLabels = loadLabels("./mnist_data/t10k-labels-idx1-ubyte")

func imageToRuneBox(image: array[28, array[28, uint8]]): seq[seq[Rune]] =
    result = newSeq[seq[Rune]](28)
    for h in 0..<28:
        result[h] = newSeq[Rune](28)
        for w in 0..<28:
            result[h][w] = if image[h][w] > uint8.high div 2: "█".toRune else: " ".toRune


var fb = newFramebuffer()

fb.add(
    trainImages[^1].imageToRuneBox,
    0,0
)

fb.print()