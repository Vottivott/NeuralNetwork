import numpy as np
import Image
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    data = 1.0 - data / 255.0 # Make black pixels 1 and white 0
    return data

# def save_image( npdata, outfilename ) :
#     img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
#     img.save( outfilename )

data = load_image("HannesNumbers\Hannes\digit6.png")

import network

neural_net = network.Network()
neural_net.load()
classified_digit = neural_net.classify(data)

def get_indefinite_article(classified_digit):
    return "an" if classified_digit == 8 else "a"

print "Handwritten digit classified as " + get_indefinite_article(classified_digit) + " " + str(classified_digit)

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.imshow(1-data, cmap = cm.Greys_r)
plt.show()