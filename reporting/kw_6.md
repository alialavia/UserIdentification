# 13.02 - 17.02

## Work Done

**ToDos:**
- [] Fix tracking safety measure
- [x] Request Handler: Implement priority queue (for ID-requests and profile picture updates)
- [x] Bugfix: Profile picture of 2nd person not displayed (in case he leaves right after initialization)
- [ ] Influence of background
- [ ] Test network recording
- [ ] Schedule demo scene
- [ ] Refactor One-VS-Rest embeddings model (use general model dir)
- [ ] Extend UserManager API
- [ ] Cascaded identification for critical cases (e.g. metric learning or pick the one with higher ABOD weighting)
	- Opt. 1: Metric Learning
	- Opt. 2: Pick "Inlier"-Class with highest relative weighting
- [ ] Feature: Identification on classifier subset - (dont evaluate on identified users in the scene)
	- Classify on subset (send ids to exclude in identification request)
	- Same for robust updates: Exclude identified users in scene which have "secure" tracking and faces are beeing recognized (SDK, possible object detections)
	

## Notes/Remarks
	
**Improvements**

- Use Kinect API for facial feature detection (dlib slows system down, cant use it in debug mode)
- Logger for client side
- Server side logger: Dump logs to hd (for evaluation purposes)

**Ideas**
- If two classifiers are too similar (cosine dist. of mean): Specific metric learning for comparison
- Don't take pictures, when user has mouth open (big influence on embedding)

## Challenges/Problems

## Literature/Personal Notes
