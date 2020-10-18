# EDA: visualize information gain via Gini Impurity

Function than can be to visualise categorical/continuous variables and rank them according to Information Gain for a split (i.e. as in the first step building a classification tree). See jupyter notebook included in repo for an example using a healthcare dataset.

Gini Impurity is calculated according to the formula

<img src="https://render.githubusercontent.com/render/math?math=\textit{Gini}: \mathit{Gini}(E) = 1 - \sum_{j=1}^{c}p_j^2">

For continuous features all possible theresholds are evaluated and the max information gain is selected (defined as the gini impurity before the split minus the weighted gini impurity of the right and left node).

The function 

```
threshold,ig = gini_calc(data,y)

```
calculates the maximum information gain for a binary split for the feature 'data' and target 'y'. Supports multiple targets (i.e. three classes 0,1,2) but only binary splits.

The function 

```
plot_gini_hist(data,y,threshold,ig,'Target','Feature',{'0':'No', '1':'Yes'})

```
plots histograms before and after the split.
    
    
## Examples on Random data

For an example on real data: visualize the interative notebook on [nbviewer here](https://nbviewer.jupyter.org/github/omadios/logistics_CVRP/blob/master/Capacitated_vehicle_routing_problem_github.ipynb)

```
length=500
s1 = np.random.normal(9, 2, length)
s2 = np.random.chisquare(10, length)+4
data =np.hstack([s1,s2])
y= np.hstack([np.ones(int(length)),np.zeros(int(length))])

threshold,ig = gini_calc(data,y)
plot_gini_hist(data,y,threshold,ig,'Target')

```

![Architecture](random_overlap.png)
|:--:|
| Example visualization of two distributions which overlap|


```
length=500
s1 = np.random.normal(9, 2, length)
s2 = np.random.normal(22, 2, length)
data =np.hstack([s1,s2])
y= np.hstack([np.ones(int(length)),np.zeros(int(length))])

threshold,ig = gini_calc(data,y)
plot_gini_hist(data,y,threshold,ig,'Target')

```


![Architecture](random_no_overlap.png)
|:--:|
| When distributions do not overlap there exist a split for which Gini Impurity is max i.e. 0.5 |







