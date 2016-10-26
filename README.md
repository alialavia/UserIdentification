# User Identification

- Automated Build of the identification node on DockerHub: https://hub.docker.com/r/matbloch/user_identification_node/



## Installation


**Setup Identification Node in Docker:**
```bash
# download image
docker pull matbloch/user_identification_node
# list images
docker images
# start container with bash
winpty docker run -ti matbloch/user_identification_node //bin/bash
```



Please visit the WIKI for the detailed instructions.

## Work Packages

**Literature Study**
-	Evaluation of anthropomorphic features in Computer Vision
-	Probabilistic clustering/identification
-	State of the art (visual) identification systems
-	Computer Vision supported collaboration systems

**Software**
-	Kinect Streaming Interface
-	Recording/Playback of sensor data
-	Base application implementation scheme (detection, classification/description, identification)
-	Sensor calibration application (masking, parameter tuning, interest point definition)
-	User feature descriptors implementation
-	Session based probabilistic identification
-	Offline identification database
-	Evaluation and handling of environmental influences
-	Demo Applications

**User Study**
-	Sensor setup/positioning
-	Robustness/Performance (different people, lighting)
-	Usability



