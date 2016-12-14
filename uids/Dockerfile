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

# temporary development addin - remove upon deployment
ENV PYTHONPATH $PYTHONPATH:/uids_tmp

# install requirements
RUN cd /uids && \
	pip2 install -r requirements.txt
	
# inform about port listening (use -p when you start the container)
EXPOSE 8080

# download detection models
CMD /bin/bash -l -c '/uids/models/get-models.sh'

# Set the default directory where CMD will execute
WORKDIR /uids

# Start server with custom port listening
# CMD ["python", "start_detector.py", "-p 8080"]