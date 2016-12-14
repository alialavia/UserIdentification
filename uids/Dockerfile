# base image: bamos/openface
FROM bamos/openface

# File Author / Maintainer
MAINTAINER Matthias Bloch

############################################################
# Dockerfile to build Identification Node Container
############################################################

# sudo add-apt-repository python-opencv
	
# Copy the application folder inside the container
ADD . /uids

ENV PYTHONPATH $PYTHONPATH:/uids

# install requirements
RUN cd /root/user_identification && \
	pip2 install -r requirements.txt
	
# inform about port listening (use -p when you start the container)
EXPOSE 8080

# download detection models
CMD /bin/bash -l -c '/root/user_identification/models/get-models.sh'

# Set the default directory where CMD will execute
WORKDIR /root/user_identification/src

# Start server with custom port listening
# CMD ["python", "start_detector.py", "-p 8080"]