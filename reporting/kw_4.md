# 30.01 - 03.02

## Work Done

- Incremental Cluster Representation for Angle Based Outlier Detection:
	- Idea: Include Points outside of Convex Hull in Subspace, Remove Points from inside Convex Hull in Subspace
	- If memory exceeded: Perform unrefinement process - discarge sampling directions with lowest variance contribution (Reduce subspace of hull calculation)
	- Additional cleanup: Remove local clustering on hull using KNN distance comparison

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
- Hull filtering does not guarantee limited cluster size (workarounds: Reduce dimensionality to simplify convex hull, KNN filtering)

## Literature/Personal Notes

- [Simplify Convex Hull](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
- [ABOD improvement](http://link.springer.com/chapter/10.1007/978-3-319-13359-1_34)
- [Continuous Angle-based Outlier Detection on High-dimensional Data Streams](http://dl.acm.org/citation.cfm?id=2790775)
- [Another ABOD method](http://www.itu.dk/people/pagh/papers/outlier.pdf)
- [ABOD discussion](https://pdfs.semanticscholar.org/b56f/31b8af2f657bf1d98c2f4679353dbca71e69.pdf)
- [Cluster Hull: A Technique for Summarizing Spatial Data Streams](https://www.cs.ucsb.edu/~suri/psdir/icde06.pdf)


### Ensemble Classification

- Weighted Voting (e.g. classifiers weighted by 1/variance)
- Bayesian Voting

