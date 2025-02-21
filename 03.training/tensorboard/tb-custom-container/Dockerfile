
# Specifies base image and tag
FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest
RUN pip install tensorflow-datasets
WORKDIR /root

# Installs additional packages as you need.

# Copies the trainer code to the docker image.
COPY task.py /root/task.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
