### image processing
# install.packages("magick")
# install.packages('rsvg')
# install.packages('imager')
library(magick)
library(rsvg)
library(imager)

sunflower = image_read('24459548_27a783feda.jpg')
print(sunflower)


## imager way of reading in the image
sunflower = load.image('24459548_27a783feda.jpg')
dim(sunflower)
plot(sunflower)

gs = grayscale(sunflower)
plot(gs)
dim(gs)


