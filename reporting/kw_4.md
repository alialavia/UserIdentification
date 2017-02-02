# 30.01 - 03.02

## Work Done

## Notes/Remarks

**Improvements**
- How to make decision from multiple sample classifications (Variance values)
	- Sample weighting based on Variance Value (samples with value around threshold are less safe) - weighted voting
- Sample Quality
	- Blur filter: Blur Metrics
	- Scene Illumination: LQ-Index (Luminance Quality Index)
	
	
**Experiments**
- Feature Importance
	- Visualize output of CNN Layers
	- Measure Classification Error increase when disabling individual features
	- PCA Eigenvalues hint feature weighting
- Visualize Variance Influence
	- Compare distance of fotos to assumed "optimal" front view
	- Display outliers
	
**Problems to deal with**
- ID changes during update (tracker id changes)
- Tracker looses person (tracks object)
- Client side: resize, then do facial detection
	
	
## Challenges/Problems

- Cost of calculation of Convex Hull increases exponentially with number of dimensions

## Literature/Personal Notes


### Ensemble Classification

- Weighted Voting (e.g. classifiers weighted by 1/variance)
- Bayesian Voting

