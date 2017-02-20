# 13.02 - 17.02

## Work Done

**ToDos:**
- [x] Fix tracking safety measure
- [x] Request Handler: Implement priority queue (for ID-requests and profile picture updates)
- [x] Bugfix: Profile picture of 2nd person not displayed (in case he leaves right after initialization)
- [x] Test network recording
	- [x] Network storage ordered: Both network and USB3-Hub Streaming suffer from severe lag (approx. 3fps) when buffer is written to disk

- [x] Cascaded identification for critical cases (e.g. metric learning or pick the one with higher ABOD weighting)
	- [ ] Opt. 1: Metric Learning (Large Margin Nearest Neighbor)
		- Tested 30vs30 samples approx. 3.6 sec - huge positive impact on cosineDist and ABOD thresholding
	- [x] Opt. 2: Pick "Inlier"-Class with lowest mean cosine distance - working OK in most cases but not for degraded ClusterHull model
- [x] Meanshift-Cluster: Keep K-Nearest points around moving average (agglomerate till max size is reached and cluster is more stable)

--------
- [ ] Feature: Identification on classifier subset - (dont evaluate on identified users in the scene)
	- Classify on subset (send ids to exclude in identification request)
	- Same for robust updates: Exclude identified users in scene which have "secure" tracking and faces are beeing recognized (SDK, possible object detections)
- [ ] Refactor One-VS-Rest embeddings model (use general model dir)
- [ ] Influence of background
- [ ] Extend UserManager API
- [ ] Schedule demo scene

## Notes/Remarks
	
**Improvements**

- [ ] Use Kinect API for facial feature detection (dlib slows system down, cant use it in debug mode)
- [ ] Logger for client side
- [ ] Server side logger: Dump logs to hd (for evaluation purposes)
- [ ] Dlib Aligner: Use Kinect Face BB to initialize landmark detector

**Ideas**
- If two classifiers are too similar (cosine dist. of mean): Specific metric learning for comparison
- Don't take pictures, when user has mouth open (big influence on embedding)


## Challenges/Problems

**Problems**
- Hull Model (Include new points outside, clean inside, KNN thinning) degrades model (better classification if only first 40 samples are kept as model)
	- Both with KNN removal enabled/disable
	- Hull inversion does not help (even worse)
	
**Solutions/Ideas**
- Model class data as multiple convex hall layers instead of a single one to increase stability
- Incremental(?) GMM with random sampling for ABOD calculation
- [x] K-Nearest points around moving average (agglomerate till max size is reached and cluster is more stable)
	
## Literature/Personal Notes
- [ClusterHull](https://www.cs.ucsb.edu/~suri/psdir/icde06.pdf)