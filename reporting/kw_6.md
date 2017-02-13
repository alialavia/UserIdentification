# 13.02 - 17.02

## Work Done


## Notes/Remarks

**ToDos:**
- Fix tracking safety measure
- Request Handler: Implement priority queue (for ID-requests and profile picture updates)
- Bugfix: Profile picture of 2nd person not displayed (in case he leaves right after initialization)
- Influence of background
- Test network recording
- Schedule demo scene
- Refactor One-VS-Rest embeddings model (use general model dir)
- Extend UserManager API

**Improvements**
- Use Kinect API for facial feature detection (dlib slows system down, cant use it in debug mode)
- Logger for client side
- Server side logger: Dump logs to hd (for evaluation purposes)

**Ideas**
- If two classifiers are too similar (cosine dist. of mean): Specific metric learning for comparison
- Don't take pictures, when user has mouth open (big influence on embedding)

## Challenges/Problems

## Literature/Personal Notes