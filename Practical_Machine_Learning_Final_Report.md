Practical Machine Learning Prediction Assignment
================================================

This file was produced during a homework assignment of Coursera's MOOC <b>Practical Machine Learning</b> from <b>Johns Hopkins Bloomberg School of Public Health</b>.

Programmer: <b>Ravijeet</b>

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [<http://groupware.les.inf.puc-rio.br/har>](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

Data Source
-----------

The training data for this project is available here:
[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)
The test data is available here:
[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)
The data for this project comes from this original source: [<http://groupware.les.inf.puc-rio.br/har>](http://groupware.les.inf.puc-rio.br/har).

Goal
----

The goal of this project is to predict the manner in which the enthusiasts did the exercise. There is a "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.
1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to \< 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

Libraries installed
-------------------

``` r
library(Hmisc)
```

    ## Loading required package: lattice

    ## Loading required package: survival

    ## Loading required package: Formula

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'Hmisc'

    ## The following objects are masked from 'package:base':
    ## 
    ##     format.pval, round.POSIXt, trunc.POSIXt, units

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.2.5

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:survival':
    ## 
    ##     cluster

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.2.5

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:Hmisc':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(foreach)
library(doParallel)
```

    ## Loading required package: iterators

    ## Loading required package: parallel

``` r
set.seed(2048)
options(warn=-1)
```

Getting Data
------------

Firstly,setting current working directory.

``` r
setwd("~/Project_related/GITHUB")
```

Loading the data both : training and test data and replacing the "\#DIV/0!" values with NA

``` r
training_data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
evaluation_data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
```

### Preparing data

Converting columns 8 to end as numeric

``` r
for(i in c(8:ncol(training_data)-1)) {training_data[,i] = as.numeric(as.character(training_data[,i]))}

for(i in c(8:ncol(evaluation_data)-1)) {evaluation_data[,i] = as.numeric(as.character(evaluation_data[,i]))}
```

Modelling
---------

I noticed that there are some columns which are blank are not contributing to the prediction. So, selected a feature set that only included complete columns.I also ignored user name, timestamps and windows.
Determining and displaying out feature set

``` r
feature_set <- colnames(training_data[colSums(is.na(training_data)) == 0])[-(1:7)]
model_data <- training_data[feature_set]
feature_set
```

    ##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
    ##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
    ##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
    ## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
    ## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
    ## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
    ## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
    ## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
    ## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
    ## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
    ## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
    ## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
    ## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
    ## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
    ## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
    ## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    ## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    ## [52] "magnet_forearm_z"     "classe"

Now, we have the model data built from our feature set.

``` r
idx <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[idx,]
testing <- model_data[-idx,]
```

We will now build 5 random forests with 150 trees each. To build this model we can use parallel processing

``` r
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest')%dopar%{
randomForest(x, y, ntree=ntree) 
}
```

Error reports
-------------

Provide error reports for both training and test data.

``` r
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4185    0    0    0    0
    ##          B    0 2848    0    0    0
    ##          C    0    0 2567    0    0
    ##          D    0    0    0 2412    0
    ##          E    0    0    0    0 2706
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9997, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

``` r
predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    5    0    0    0
    ##          B    0  941    6    0    0
    ##          C    0    3  848    8    2
    ##          D    0    0    1  796    1
    ##          E    0    0    0    0  898
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9947          
    ##                  95% CI : (0.9922, 0.9965)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9933          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9916   0.9918   0.9900   0.9967
    ## Specificity            0.9986   0.9985   0.9968   0.9995   1.0000
    ## Pos Pred Value         0.9964   0.9937   0.9849   0.9975   1.0000
    ## Neg Pred Value         1.0000   0.9980   0.9983   0.9981   0.9993
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1919   0.1729   0.1623   0.1831
    ## Detection Prevalence   0.2855   0.1931   0.1756   0.1627   0.1831
    ## Balanced Accuracy      0.9993   0.9950   0.9943   0.9948   0.9983

Conclusions and Test Data Submit
--------------------------------

It is very clear from the confusion matrix that this model is very accurate.My test data was around 99% accurate and as expected nearly all of the submitted test cases turend to be correct.

Prepare the submission

``` r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


x <- evaluation_data
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

``` r
pml_write_files(answers)
```
