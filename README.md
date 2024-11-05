## ML workshop

1. What is machine learning? 
2. Supervised vs Unsupervised 
3. When to use which model
4. What is the ML lifecycle?
5. Examples

### High level model creation flow chart

- There are many more details but this is a good workflow to follow:

```mermaid
flowchart TD
    id1[/Get Data/] --> id2[Preprocessing on all data e.g. change strings to integers]
    id2 --> id3[/Train Test split/]
    id3 --> id4[Perform feature engineering on training data only! fe.fit & fe.transform]
    id4 --> id5[/Define model and fit on training data only! m.fit/]
    id5 --> id6[Transform test data using fit feature engineering instance. Only fe.transform]
    id6 --> id7[/Make train and test predictions/]
    id7 --> id8[Use applicable metrics to score the model]
    id8 --tweak & repeat --> id4 
```

### With production and live models:


![ml_workflow](./images/mlworkflow.png)


