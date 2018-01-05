# Project: Create a Tableau Story

## Summary
The sinking of the RMS Titanic was most tragic event in the history of ship disasters. One of the reason was that, there were no enough life boats. There was a bit luck in surviving the tragedy. This visualization just helps to find out, what factors made some group people were more likely to survive than the others.

## Design Rev_0 [Beginning]

There is one outcome/dependent variable 'survival' and nine predictor/independent variables as listed below. I have created new variable named 'age_groups' based on the age of the passenger, so there are total 10 predictor variables. There are so many null values in the Age column, so I replaced those values with the median of the 'Age' column.

1. survival	(Survival	0 = No, 1 = Yes)
2. pclass	(Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd)
3. sex	(Sex)
4. Age	(Age in years)
5. sibsp	(# of siblings / spouses aboard the Titanic)
6. parch	(# of parents / children aboard the Titanic)
7. ticket	(Ticket number)
8. fare	(Passenger fare)
9. cabin	(Cabin number)
10. embarked	(Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton)
11. age_groups [ child (< 10 years), teen (10 to 19), young_adult (20 to 29), middle_aged (30 to 49), elder (> 49)]

#### Choice of Charts

The objective of this visualization to demonstrate what sort of people were likely to survive the Titanic accident. Based on data type, dimension, and relation I have decided to use following chart types.

| Variables    | Data Type     | Dimension |Chart Type |
|:------------:|:-------------:|:---------:|:---------:|
| survival      | nominal   | 1D        | bar chart |
| survival vs sex| nominal vs nominal   | 2D        | bar chart |
| survival vs pclass| nominal vs ordinal   | 2D        | bar chart |
| survival vs embarked| nominal vs ordinal   | 2D        | bar chart |
| survival vs age_groups| nominal vs ordinal   | 2D        | bar chart |
| Age           | Continuous    | 1D        | histogram |
| Fare| Continuous|1D|histogram|
|Fare vs survival| Continuous vs nominal| 2D| bar chart|

Considering the nature or the type of the data I have heavily used bar charts and histograms.
I have decided to use soft and natural colours. Encoding of the data as mainly done using the length of the bars. Based on need and requirement the horizontal and vertical bar charts are selected.   

[Link to Project Rev_0](https://public.tableau.com/profile/pradipmore#!/vizhome/titanic_83/TitanicStory?publish=yes)


## Feedback

### Feedback #1

1. What is C, Q, S labels?
2. What is Pclass on the first slide?
3. On slide#1 user could not understand the some information shown on the tooltip.
4. the filter on "Sex" needs to be applied on entire worksheet.
5. on third slide Age_transformed
6. on slide#3 and the bar chart survival with respect to age_groups the legend 'Survived or Died' is not required
7. In slide#4 reader could not understand the column of siblings or spouse

### Feedback #2

1. How siblings and spouse survival  data is divided  in 8 rows ?
2. Similar question I have about parents and child data?
3. In process of embankment why data is divided in to C , Q and S format?

## Design Rev_01 [After the Feedback]


1. Crated another mapping variable 'port of embarkation' which maps the 'C', 'Q' and 'S' to actual city names like "C = Cherbourg, Q = Queenstown, S = Southampton"

2. Different filters added on slide #1 so that user can explore the data by it's own.

3. To create space and enhance the visualization multiple filters were created with dropdown menu.

4. On slide#1 all filters were applied to all worksheet using that data source, so that changes can happen to all plot depending upon the selection of different filters.

5. The unnecessary legends have been removed.

[Link to Project Rev_01](https://public.tableau.com/profile/pradipmore#!/vizhome/titanic_Rev_01/TitanicStory)

## Resources

1. [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
2. [http://kb.tableau.com/articles/Issue/internal-error-occuring-when-publishing-a-viz-to-tableau-public](http://kb.tableau.com/articles/Issue/internal-error-occuring-when-publishing-a-viz-to-tableau-public)
3. [http://onlinehelp.tableau.com/current/pro/desktop/en-us/functions_functions_logical.html](http://onlinehelp.tableau.com/current/pro/desktop/en-us/functions_functions_logical.html)
3. [https://onlinehelp.tableau.com/current/pro/desktop/en-us/publish_workbooks_tableaupublic.html](https://onlinehelp.tableau.com/current/pro/desktop/en-us/publish_workbooks_tableaupublic.html)
4. [http://www.perceptualedge.com/articles/ie/the_right_graph.pdf](http://www.perceptualedge.com/articles/ie/the_right_graph.pdf)
5. [https://solomonmessing.wordpress.com/2014/10/11/when-to-use-stacked-barcharts/](https://solomonmessing.wordpress.com/2014/10/11/when-to-use-stacked-barcharts/)
6. [http://www-ist.massey.ac.nz/dstirlin/CAST/CAST/Hstructures/structures_c2.html](http://www-ist.massey.ac.nz/dstirlin/CAST/CAST/Hstructures/structures_c2.html)
