## the following details the package requirements and system setup needed for
## both networks and supporting functions

# for R2N2
numpy
pygpu
scikit-learn
## if theano version is newer gpu backend will not function.
theano=1.0.2
easydict (pip install this one)

# for 3D-RecGAN
numpy
scikit-learn
matplotlib
tensorflow-gpu
