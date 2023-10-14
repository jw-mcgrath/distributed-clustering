# Use the official Ray base image
FROM rayproject/ray:latest-aarch64
# todo create an anaconda environment
# Should already be avilable in the base image
RUN ["conda", "install", "scikit-learn"]

# Set the working directory in the container
WORKDIR /app
