# Use the TensorFlow notebook as the base image from quay.io
FROM quay.io/jupyter/tensorflow-notebook:cuda-latest


# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install librosa
RUN pip install soundfile
RUN pip install pydub
RUN conda install -y portaudio
RUN pip install pyAudio
RUN pip install simpleaudio
RUN pip install audioread
RUN conda install -y libsndfile portmidi
USER root
RUN apt-get update && apt-get install -y liblo-dev
USER jovyan
RUN pip install pyo

# Install any additional dependencies
# RUN pip install <other-dependencies>

# Set the working directory
WORKDIR /home/jovyan/work

# Expose necessary ports (if any)
EXPOSE 8888