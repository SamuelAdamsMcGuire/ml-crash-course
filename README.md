## ML workshop

1. What is machine learning? 
2. Supervised vs Unsupervised 
3. When to use which model
4. What is the ML lifecycle?
5. Examples
6. MLFlow

### High level model creation flow chart

```mermaid
 flowchart TD
    id1[/Define Goal X,y $$$/] --> id2[Get Data]
    id2 --> id3[/Clean and make data tabular/]
    id3 --> id4[Train - Test split]
    id4 --> id13[Training data]
    id13 --> id5[/EDA  Exploratory Data Analysis/]
    id5 --> id6[Feature Engineering]
    id6 --> id7[/Train model on training data only/]
    id7 --> id8[Hyperparameter Optimization]
    id8 --> id9(Evaluate model using test data)
    id9 --> id10{Deploy and Monitor}
    id10 --> id15(Learn and Improve)
    id15 --> id1
    id4 --> id12[test data]
    id12 --> id14[transform based on training data]
    id14 --> id9
```


![ml_workflow](./images/mlworkflow.png)


