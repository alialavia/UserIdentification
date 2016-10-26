# base image: python2:slim
FROM python:2-slim

# File Author / Maintainer
MAINTAINER Matthias Bloch

############################################################
# Dockerfile to build Identification Node Container
############################################################


# Install app dependencies
RUN pip install --no-cache-dir numpy && \
	apt-get install -y python-opencv

# Copy the application folder inside the container
ADD . /root/user_identification

# inform about port listening (use -p when you start the container)
EXPOSE 80:8080

# Set the default directory where CMD will execute
WORKDIR /root/user_identification/src

# Start server

# with custom port listening
# CMD ["python", "start_detector.py", "-p 8000"]


RUN echo 'We are ready to go! Lets start the server...'

# CMD python ./core.py

# CMD /bin/bash -l -c 'start_detector.sh'