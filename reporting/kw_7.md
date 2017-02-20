# 20.02 - 24.02

## Work Done

**ToDos:**

- [x] Extend UserManager API

------

- [ ] Detect when tracking has switched to static object
- [ ] Tracking consistency check using skeleton and positional data
- [ ] Detect inconsitent incomming updates (when user tracking has switched without triggering the "unsafe" tracking state)
	- e.g. two distinct clusters with larger inter-cluster distance and small intra-cluster distance
	- action: drop update (+trigger reidentification)
- [ ] Allow to have differently partitioned face grids for updates and identification
- [ ] Feature: Identification on classifier subset - (dont evaluate on identified users in the scene)
	- Classify on subset (send ids to exclude in identification request)
	- Same for robust updates: Exclude identified users in scene which have "secure" tracking and faces are beeing recognized (SDK, possible object detections)
- [ ] Refactor One-VS-Rest embeddings model (use general model dir)
- [ ] Influence of background
- [ ] Schedule demo scene

## Notes/Remarks

## Challenges/Problems

## Literature/Personal Notes
