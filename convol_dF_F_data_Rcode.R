
library(FastLZeroSpikeInference)
library("data.table")

# Read data
#dF_F_data <- read.csv("dF_F_data.csv", header = FALSE) # much slower than fread
setwd("//research.files.med.harvard.edu/Neurobio/Fishell Lab/Shuhan_HMS folder/NGFC/2P/NGFC/Jam2Cre/NP14D_ROI1_020922/NP14D_ROI1_image1")
dF_F_data <- fread("dF_F_data.csv")

# FastLZeroSpikeInference
fit <- list()
spikes <- list()
estimated_calcium <- list()
for (i in 1:nrow(dF_F_data)) {                              
  fit[[i]] <- estimate_spikes(dat = dF_F_data[i,], gam = 0.95, lambda = 1, constraint = T, estimate_calcium = T)
  spikes[[i]] <- fit[[i]]$spikes
  estimated_calcium[[i]] <- fit[[i]]$estimated_calcium
}


# save spikes without index
capture.output(cat(format(spikes), fill = getOption("width")), file = "spikes.txt")
capture.output(cat(format(estimated_calcium), fill = getOption("width")), file = "estimated_calcium.txt")
