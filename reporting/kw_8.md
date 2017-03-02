# 27.02 - 3.03

## Work Done


## Notes/Remarks

**Possible Improvements**

- Test Update Batch with Local Outlier Factor (new SKLearn implementation) or Clustering for inconsistency
- Inter-Class metric learning, when clusters become too similar (Fowlkes Mallows Index, Calinski and Harabaz score)
- Template Face Landmark coordinates for Kinect SDK Bounding Box (Kinect face detection is however jittery)

**Todo**
- Tune ABOD Threshold: Regression against LFW dataset or two very similar faces (large amount of pictures needed)
- Evaluate feature activation in case two features are very similar (how can we differentiate best between these faces)
	- idea: outlier detection in subspace, if very similar
	- feature selection: (incremental-)PCA, Chi-Squared Test etc. [SKLearn tutorial](http://machinelearningmastery.com/feature-selection-machine-learning-python/)

- Avoid too much drift in mean-shift data cluster
	
## Challenges/Problems

## Literature/Personal Notes

- Comparison of Manifold Learning Methods (SKlearn): [SKlearn](http://scikit-learn.org/dev/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py)