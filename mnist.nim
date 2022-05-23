type
    MNISTImage = object
        pixels: array[28*28, uint8]
        label: int


func loadImages(imageFilename, labelFilename: string): seq[MNISTImage] =
    discard