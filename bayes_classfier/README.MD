# 朴素贝叶斯

下载[数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data)，然后以.csv扩展名保存（如：pima-indians-diabetes.data.csv）。查看[数据集中属性的描述](https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.names)。

For Each Attribute: (all numeric-valued)
```
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)
```
**算法步骤**
1. 提取数据特征
```
   按类别划分数据
   计算均值
   计算标准差
   提取数据集特征
   按类别提取属性特征
```

2. 预测 
```
    计算高斯概率密度函数
    计算对应类的概率
    单一预测
    评估精度
```




