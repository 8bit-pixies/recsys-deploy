This folder contains the same models trained under `/notebooks` but now with some variation of precision metrics

**How this works**

*  Perform a random data split
*  In the test set, spawn additional instances which drop a label
*  Train the model in-memory
*  Report the MAP@k (k chosen to be 5)

