# Ensemble Learning
Using the classes are quite simple. If you'd like to run the experiement, simply use the run.sh file. For the 3 ensemble methods, each has its own corresponding class.

## Adaboosted Trees and Bagged Trees
To use the adaboosted tree, simply create a tree using the constructor. The constructor takes an error function, number of trees to create and a depth. After, if you'd like to train the trees, then simply use the train_dataset function. The train_dataset function takes in 3 arguments, examples, attributes and labels. This function simply returns nothing. Following, if you'd like to test the new tree, then use the test_dataset function which takes examples and labels. The test_dataset function returns the final predictions and the error.

## Random Forest
To use the random forest, simply create a tree using the constructor. The constructor takes an error function, the number of random features to split, number of trees to create and a depth. After, if you'd like to train the  random forest, then simply use the train_dataset function. The train_dataset function takes in 3 arguments, examples, attributes and labels. This function simply returns nothing. Following, if you'd like to test the new tree, then use the test_dataset function which takes examples and labels. The test_dataset function returns the final predictions and the error.