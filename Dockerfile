# Use the base image
FROM nvcr.io/nvidia/tritonserver:23.06-py3

# Install the required Python packages
RUN pip install optimum==1.9.0 onnxruntime==1.15.1 onnx==1.14.0
