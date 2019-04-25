# install.packages("magick")
# install.packages('rsvg')
# install.packages('imager')


### 1. image processing
library(magick)
library(rsvg)
library(imager)

# magick way of readin in the image
sunflower = image_read('24459548_27a783feda.jpg')
print(sunflower)

# imager way of reading in the image
sunflower = load.image('24459548_27a783feda.jpg')
dim(sunflower)
plot(sunflower)

# turn it into a grayscale image 
gs = grayscale(sunflower)
plot(gs)
dim(gs)

gs_vec = c()
for (i in 1:dim(gs)[2]){
  gs_vec = cbind(gs_vec,t(gs[,i]))
}
rm(i)
dim(gs_vec)
hist(gs_vec)

