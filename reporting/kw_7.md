# 20.02 - 24.02

## Work Done

**ToDos:**

- [x] Extend UserManager API
- [x] Detect when tracking has switched to static object
	- [x] Movement detections
	- [x] Face detections
	- [ ] Optical flow (implemented yet not integrated)
- [x] Only take profile picture when user has eyes open and mouth closed
	- Condition too strong (does take too long to find a good picture)
- [x] Parallel face detection in case Kinect fails
	- Threaded parallel Face Detection (HoG). Would slow system down to appox 1fps if detection is done in realtime.
	
------

- [ ] Object detection using angle between spine and head (for object usually twisted and fast moving)
- [ ] Tracking consistency check using skeleton and positional data
- [ ] Detect inconsitent incomming updates (when user tracking has switched without triggering the "unsafe" tracking state)
	- e.g. two distinct clusters with larger inter-cluster distance and small intra-cluster distance (K-Means Clustering)
	- action: drop update (+trigger reidentification)
- [ ] Allow to have differently partitioned face grids for updates and identification
- [ ] Feature: Identification on classifier subset - (dont evaluate on identified users in the scene)
	- Classify on subset (send ids to exclude in identification request)
	- Same for robust updates: Exclude identified users in scene which have "secure" tracking and faces are beeing recognized (SDK, possible object detections)
	- Problems: If tracking has switched or innvalid identification: Prevents forced reinitialization on duplicate ID
- [ ] Refactor One-VS-Rest embeddings model (use general model dir)
- [ ] Influence of background
- [ ] Schedule demo scene


------

- [ ] Force identification using less samples on timeout
- [ ] Add categoric "live" confience measure
- [ ] Streamlining classification/identification/updates (lot of work, need to remodel server and client)


**Debug:**
- [ ] Sinze object detection: Random crash (without Exception) when user leaves scene

## Notes/Remarks

- Focusing on Report in the comming weeks

## Challenges/Problems

## Literature/Personal Notes
