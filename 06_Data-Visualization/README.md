# 6_ Data Visualization

Open .R scripts in Rstudio for line-by-line execution.

See [10_ Toolbox/3_ R, Rstudio, Rattle](https://github.com/MrMimic/data-scientist-roadmap/tree/master/10_Toolbox#3_-r-rstudio-rattle) for installation.

## 1_ Data exploration in R

In mathematics, the graph of a function f is the collection of all ordered pairs (x, f(x)). If the function input x is a scalar, the graph is a two-dimensional graph, and for a continuous function is a curve. If the function input x is an ordered pair (x1, x2) of real numbers, the graph is the collection of all ordered triples (x1, x2, f(x1, x2)), and for a continuous function is a surface.

## 2_ Uni, bi and multivariate viz

### Univariate

The term is commonly used in statistics to distinguish a distribution of one variable from a distribution of several variables, although it can be applied in other ways as well. For example, univariate data are composed of a single scalar component. In time series analysis, the term is applied with a whole time series as the object referred to: thus a univariate time series refers to the set of values over time of a single quantity.

### Bivariate

Bivariate analysis is one of the simplest forms of quantitative (statistical) analysis.[1] It involves the analysis of two variables (often denoted as X, Y), for the purpose of determining the empirical relationship between them.

### Multivariate

Multivariate analysis (MVA) is based on the statistical principle of multivariate statistics, which involves observation and analysis of more than one statistical outcome variable at a time. In design and analysis, the technique is used to perform trade studies across multiple dimensions while taking into account the effects of all variables on the responses of interest.

## 3_ ggplot2

### About

ggplot2 is a plotting system for R, based on the grammar of graphics, which tries to take the good parts of base and lattice graphics and none of the bad parts. It takes care of many of the fiddly details that make plotting a hassle (like drawing legends) as well as providing a powerful model of graphics that makes it easy to produce complex multi-layered graphics.

[http://ggplot2.org/](http://ggplot2.org/)

### Documentation

### Examples

[http://r4stats.com/examples/graphics-ggplot2/](http://r4stats.com/examples/graphics-ggplot2/)

## 4_ Histogram and pie (Uni)

### About

Histograms and pie are 2 types of graphes used to visualize frequencies. 

Histogram is showing the distribution of these frequencies over classes, and pie the relative proportion of this frequencies in a 100% circle.

## 5_ Tree & tree map

### About

[Treemaps](https://en.wikipedia.org/wiki/Treemapping) display hierarchical (tree-structured) data as a set of nested rectangles.
Each branch of the tree is given a rectangle, which is then tiled with smaller rectangles representing sub-branches.
A leaf node’s rectangle has an area proportional to a specified dimension of the data.
Often the leaf nodes are colored to show a separate dimension of the data.

### When to use it ?

- Less than 10 branches.
- Positive values.
- Space for visualisation is limited.

### Example

![treemap-example](https://jingwen-z.github.io/images/20181030-treemap.png)

This treemap describes volume for each product universe with corresponding surface. Liquid products are more sold than others.
If you want to explore more, we can go into products “liquid” and find which shelves are prefered by clients.

### More information

[Matplotlib Series 5: Treemap](https://jingwen-z.github.io/data-viz-with-matplotlib-series5-treemap/)

## 6_ Scatter plot

### About

A [scatter plot](https://en.wikipedia.org/wiki/Scatter_plot) (also called a scatter graph, scatter chart, scattergram, or scatter diagram) is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data.

### When to use it ?

Scatter plots are used when you want to show the relationship between two variables.
Scatter plots are sometimes called correlation plots because they show how two variables are correlated.

### Example

![scatter-plot-example](https://jingwen-z.github.io/images/20181025-pos-scatter-plot.png)

This plot describes the positive relation between store’s surface and its turnover(k euros), which is reasonable: for stores, the larger it is, more clients it can accept, more turnover it will generate.

### More information

[Matplotlib Series 4: Scatter plot](https://jingwen-z.github.io/data-viz-with-matplotlib-series4-scatter-plot/)

## 7_ Line chart

### About

A [line chart](https://en.wikipedia.org/wiki/Line_chart) or line graph is a type of chart which displays information as a series of data points called ‘markers’ connected by straight line segments. A line chart is often used to visualize a trend in data over intervals of time – a time series – thus the line is often drawn chronologically.

### When to use it ?

- Track changes over time.
- X-axis displays continuous variables.
- Y-axis displays measurement.

### Example

![line-chart-example](https://jingwen-z.github.io/images/20180916-line-chart.png)

Suppose that the plot above describes the turnover(k euros) of ice-cream’s sales during one year.
According to the plot, we can clearly find that the sales reach a peak in summer, then fall from autumn to winter, which is logical.

### More information

[Matplotlib Series 2: Line chart](https://jingwen-z.github.io/data-viz-with-matplotlib-series2-line-chart/)

## 8_ Spatial charts

## 9_ Survey plot

## 10_ Timeline

## 11_ Decision tree

## 12_ D3.js

### About

This is a JavaScript library, allowing you to create a huge number of different figure easily.

https://d3js.org/

    D3.js is a JavaScript library for manipulating documents based on data. 
    D3 helps you bring data to life using  HTML, SVG, and CSS. 
    D3’s emphasis on web standards gives you the full capabilities of modern browsers without tying yourself to a proprietary framework, combining powerful visualization components and a data-driven approach to DOM manipulation. 

### Examples

There is many examples of chars using D3.js on [D3's Github](https://github.com/d3/d3/wiki/Gallery).

## 13_ InfoVis

## 14_ IBM ManyEyes

## 15_ Tableau

## 16_ Venn diagram

### About

A [venn diagram](https://en.wikipedia.org/wiki/Venn_diagram) (also called primary diagram, set diagram or logic diagram) is a diagram that shows all possible logical relations between a finite collection of different sets.

### When to use it ?

Show logical relations between different groups (intersection, difference, union).

### Example

![venn-diagram-example](https://jingwen-z.github.io/images/20181106-venn2.png)

This kind of venn diagram can usually be used in retail trading.
Assuming that we need to study the popularity of cheese and red wine, and 2500 clients answered our questionnaire.
According to the diagram above, we find that among 2500 clients, 900 clients(36%) prefer cheese, 1200 clients(48%) prefer red wine, and 400 clients(16%) favor both product.

### More information

[Matplotlib Series 6: Venn diagram](https://jingwen-z.github.io/data-viz-with-matplotlib-series6-venn-diagram/)

## 17_ Area chart

### About

An [area chart](https://en.wikipedia.org/wiki/Area_chart) or area graph displays graphically quantitative data.
It is based on the line chart. The area between axis and line are commonly emphasized with colors, textures and hatchings.

### When to use it ?

Show or compare a quantitative progression over time.

### Example

![area-chart-example](https://jingwen-z.github.io/images/20181114-stacked-area-chart.png)

This stacked area chart displays the amounts’ changes in each account, their contribution to total amount (in term of value) as well.

### More information

[Matplotlib Series 7: Area chart](https://jingwen-z.github.io/data-viz-with-matplotlib-series7-area-chart/)

## 18_ Radar chart

### About

The [radar chart](https://en.wikipedia.org/wiki/Radar_chart) is a chart and/or plot that consists of a sequence of equi-angular spokes, called radii, with each spoke representing one of the variables. The data length of a spoke is proportional to the magnitude of the variable for the data point relative to the maximum magnitude of the variable across all data points. A line is drawn connecting the data values for each spoke. This gives the plot a star-like appearance and the origin of one of the popular names for this plot.

### When to use it ?

- Comparing two or more items or groups on various features or characteristics.
- Examining the relative values for a single data point.
- Displaying less than ten factors on one radar chart.

### Example

![radar-chart-example](https://jingwen-z.github.io/images/20181121-multi-radar-chart.png)

This radar chart displays the preference of 2 clients among 4.
Client c1 favors chicken and bread, and doesn’t like cheese that much.
Nevertheless, client c2 prefers cheese to other 4 products and doesn’t like beer.
We can have an interview with these 2 clients, in order to find the weakness of products which are out of preference.

### More information

[Matplotlib Series 8: Radar chart](https://jingwen-z.github.io/data-viz-with-matplotlib-series8-radar-chart/)

## 19_ Word cloud

### About

A [word cloud](https://en.wikipedia.org/wiki/Tag_cloud) (tag cloud, or weighted list in visual design) is a novelty visual representation of text data. Tags are usually single words, and the importance of each tag is shown with font size or color. This format is useful for quickly perceiving the most prominent terms and for locating a term alphabetically to determine its relative prominence.

### When to use it ?

- Depicting keyword metadata (tags) on websites.
- Delighting and provide emotional connection.

### Example

![word-cloud-example](https://jingwen-z.github.io/images/20181127-basic-word-cloud.png)

According to this word cloud, we can globally know that data science employs techniques and theories drawn from many fields within the context of mathematics, statistics, information science, and computer science. It can be used for business analysis, and called “The Sexiest Job of the 21st Century”.

### More information

[Matplotlib Series 9: Word cloud](https://jingwen-z.github.io/data-viz-with-matplotlib-series9-word-cloud/)
