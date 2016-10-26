# base image: python2:slim
FROM python:2:slim

# File Author / Maintainer
MAINTAINER Matthias Bloch

############################################################
# Dockerfile to build Identification Node Container
############################################################


# Install app dependencies
RUN pip install Flask

# Copy the application folder inside the container
ADD . /root/user_identification


# Port mapping
# private and public mapping
EXPOSE 80:8080

# Set the default directory where CMD will execute
WORKDIR /root/user_identification/src

# Execute script
# CMD python ./start_detector.py
# with port listening
# CMD ["python", "start_detector.py", "-p 8000"]

# CMD /bin/bash -l -c 'start_detector.sh'