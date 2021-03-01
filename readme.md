# 一、获取Raw Data

### 1.Gradient Data

**请求URL：** 
- ` http://10.76.2.232:8000/client_grad `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回最新一轮的gradient                                                     |

**返回示例**
- 返回最新一轮的数据
``` 
{
    'round': 1,
    'data': {
        '0': {'conv1': [1,2,3,4], 'conv2': [1,2,3,4], 'dense': [1,2,3,4]},
        '1': {'conv1': [5,6,7,8], 'conv2': [5,6,7,8], 'dense': [5,6,7,8]}
    }
}
```

- 返回指定轮的数据

``` 
{
    '0': {'conv1': [1,2,3,4], 'conv2': [1,2,3,4], 'dense': [1,2,3,4]},
    '1': {'conv1': [5,6,7,8], 'conv2': [5,6,7,8], 'dense': [5,6,7,8]}
}
```

 **返回参数说明** 

| 字典元素   | 类型 | 说明                                            |
| :------- | :--- | ----------------------------------------------- |
| key    | int  | client id                                    |
| value | dict | key = layers, value = gradient of this layer |

### 2.Averaged Gradient Data

**请求URL：** 
- ` http://10.76.2.232:8000/avg_grad `
  

**请求方式：**
- GET

**参数：**

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回最新一轮的averaged gradient  

**返回示例**
- 返回最新一轮的数据

``` 
{
    'round': 1,
    'data': {
        'conv1': [1,2,3,4],
        'conv2': [1,2,3,4],
        'dense': [1,2,3,4]
    }
}
```

- 返回指定轮的数据

``` 
{
    'conv1': [1,2,3,4],
    'conv2': [1,2,3,4],
    'dense': [1,2,3,4]
}
```

 **返回参数说明** 

| 字典元素   | 类型 | 说明                                                         |
| :------- | :--- | ------------------------------------------------------------ |
| key    | str  | layers                                                 |
| value | list | averaged gradient of this layer|

### 3.Model Accuracy and Loss

**请求URL：** 
- ` http://10.76.2.232:8000/performance `
  

**请求方式：**
- GET

**参数：**

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前为止所有轮的数据
| number      | 否   | int  | 请求从round轮开始前几轮的数据，默认返回1轮的数据



**返回示例**

- 返回所有轮的数据

```
{
    '0': {
        'train': {
              '0': {'accuracy': 0.15, 'loss': 80.15}
              '1': {'accuracy': 0.25, 'loss': 65.15}
          },
        'test': {
              '0': {'accuracy': 0.15, 'loss': 80.15}
              '1': {'accuracy': 0.25, 'loss': 65.15}
          }
	  },
    '1': {
        'train': {
              '0': {'accuracy': 0.15, 'loss': 80.15}
              '1': {'accuracy': 0.25, 'loss': 65.15}
          },
        'test': {
              '0': {'accuracy': 0.15, 'loss': 80.15}
              '1': {'accuracy': 0.25, 'loss': 65.15}
          }
    }
}
```

- 返回某一轮的数据

```
{
    'train': {
          '0': {'accuracy': 0.15, 'loss': 80.15}
          '1': {'accuracy': 0.25, 'loss': 65.15}
      },
    'test': {
          '0': {'accuracy': 0.15, 'loss': 80.15}
          '1': {'accuracy': 0.25, 'loss': 65.15}
      }
}
```

 **返回参数说明** 

| 字典元素 | 类型 | 说明                                                         |
| :----- | :--- | ------------------------------------------------------------ |
| key  | int  | 当前训练轮数                                                 |
| train  | dict | key = client id, value = accuracy and loss of this client in the training stage |
| test   | dict | key = client id, value = accuracy and loss of this client in the testing stage |


### 4.Trained Clients 

**说明** 
- 返回某些训练轮数中，参与训练的client id

**请求URL：** 
- ` http://10.76.2.232:8000/trained_clients/ `
  

**请求方式：**
- GET

**参数：**

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 是   | int  | 请求第几轮训练的数据
| number      | 是   | int  | 请求从round轮开始前几轮的数据



**返回示例**


```
{
    '0': [1,2,3,4,5],
    '1': [7,8,9,10,11]
}
```

 **返回参数说明** 

| 字典元素 | 类型 | 说明                                                         |
| :----- | :--- | ------------------------------------------------------------ |
| key  | int  | 当前训练轮数                                                 |
| value  | list  | 参与训练的client id                                                 |


### 5. Client Weights

**说明** 
- 返回client weight

**请求URL：** 
- ` http://10.76.2.232:8000/weight/ `
  

**请求方式：**
- GET

**参数：**

- 无参数


**返回示例**


```
{
    '0': 0.05,
    '1': 0.02
}
```

 **返回参数说明** 

| 字典元素 | 类型 | 说明                                                         |
| :----- | :--- | ------------------------------------------------------------ |
| key  | str  | client id                                                 |
| value  | float  | 该client的weight                                                 |

# 二、计算异常分数

### 1.Krum方法

**简要描述：** 

- 计算最新一轮各个client的异常分数
- 基于client gradient之间的距离计算：对每个client，计算其KNN范围内的gradient距离均值
- 使用cos距离
- 分数越大越异常

**请求URL：** 
- ` http://10.76.2.232:8000/anomaly/krum/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| k      | 是   | int  | knn的k                                                       |
| layers | 否   | str | 需要用哪些layer的梯度计算异常值，例如: 'conv1', 'conv2'，默认值为 'conv1' |

**返回示例**

```
{
    'round': 1,
    'data': {
        '0': 0.5,
	    '1': 0.6,
	    '2': 0.7
    }
}
```

 **返回参数说明** 

| 字典元素 | 类型  | 说明          |
| :------- | :---- | ------------- |
| key      | str   | client id     |
| value    | float | anomaly score |



### 2.Zeno方法

**简要描述：** 

- 计算最新一轮各个client的异常分数
- 基于loss函数和gradient：score = 上轮loss - 本轮loss  - p * (本轮梯度与上轮梯度的欧式距离)
- 已做反向映射，分数越大越异常


**请求URL：** 
- ` http://10.76.2.232:8000/anomaly/zeno/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                                   |
| :----- | :--- | :--- | ------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| p      | 是   | int  | 梯度在计算分值时所占的比重，在FEMNIST的case中建议选100 |
| layers | 否   | str | 需要用哪些layer的梯度计算异常值，例如: 'conv1', 'conv2'，默认值为 'conv1' |

### 3.Auror方法

**简要描述：** 

- 计算最新一轮各个client的异常分数
- 基于gradient计算：对所有client在该轮的梯度做kmean聚类，使用每个client和其所属的聚类中心之间的欧式距离衡量异常分数
- 分数越大越异常



**请求URL：** 
- ` http://10.76.2.232:8000/anomaly/auror/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| k      | 是   | int  | kmeans的k                                                    |
| layers | 否   | list | 需要用哪些layer的梯度计算异常值，例如: ['conv1', 'conv2', 'dense']，默认值为['dense'] |

### 4.Sniper方法

**简要描述：** 

- 计算最新一轮各个client的异常分数
- 基于gradient，使用graph的maximun clique计算异常分数：在maximun clique中的异常值为0，不在的异常值为1
- 类似二分类，分数为1的是异常，分数为0的是正常值


**请求URL：** 
- ` http://10.76.2.232:8000/anomaly/sniper/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型  | 说明                                                         |
| :----- | :--- | :---- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| p      | 是   | float | 所求的maximum clique至少要包含的client数量占client总数的占比，推荐0.8及以上 |
| layers | 否   | list  | 需要用哪些layer的梯度计算异常值，例如: ['conv1', 'conv2', 'dense']，默认值为['dense'] |

### 5.PCA方法

**简要描述：** 

- 计算最新一轮各个client的异常分数
- 基于gradient：使用pca对各个client的梯度数据做降维，然后用降维后的数据重构高维梯度数据，异常分数=重构梯度与原始梯度的欧式距离
- 分数越大越异常


**请求URL：** 
- ` http://10.76.2.232:8000/anomaly/pca/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型  | 说明                                                         |
| :----- | :--- | :---- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| k      | 是   | float | 使用pca降到多少维                                            |
| layers | 否   | list  | 需要用哪些layer的梯度计算异常值，例如: ['conv1', 'conv2', 'dense']，默认值为['dense'] |

# 三、计算贡献分数

### 1.基于梯度的方法

**简要描述：** 

- 计算最新一轮各个client的贡献分数
- 基于client gradient计算：对每个client，计算其本轮梯度与上一轮联邦平均梯度的距离
- 可使用两种距离度量方式：欧式距离与cos距离
- 分数越大贡献越大


**请求URL：** 
- ` http://10.76.2.232:8000/contribution/grad_diff/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                                         |
| :----- | :--- | :--- | ------------------------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| metric | 否  | str  | 使用欧式距离: 'eu', 使用cos距离: 'cos'，默认使用欧氏距离                      |
| layers | 否   | list | 需要用哪些layer的梯度计算异常值，例如: ['conv1', 'conv2', 'dense']，默认值为['dense'] |

**返回示例**

```
{
    'round': 1,
    'data': {
        '0': 0.5,
	    '1': 0.6,
	    '2': 0.7
    }
}
```


 **返回参数说明** 

| 字典元素 | 类型  | 说明               |
| :------- | :---- | ------------------ |
| key      | str   | client id          |
| value    | float | contribution score |

### 2.基于模型performance的方法

**简要描述：** 

- 计算最新一轮各个client的贡献分数
- 基于model performance计算：对每个client，计算其不参与本轮联邦平均所得模型的performance和其参与联邦平均所得模型的performance差异
- 可使用两种方式度量模型performance：accuracy和loss
- 当衡量标准为accuracy时：分数为正表示有正贡献，数值越大贡献越大；分数为负表示有负贡献，数值越大负贡献越大
- 当衡量标准被loss时：分数为正表示有正贡献，数值越大贡献越大；分数为负表示有负贡献，数值越大负贡献越大


**请求URL：** 
- ` http://10.76.2.232:8000/contribution/perf_diff/ `
  

**请求方式：**
- GET

**参数：** 

| 参数名 | 必选 | 类型 | 说明                                       |
| :----- | :--- | :--- | ------------------------------------------ |
| round      | 否   | int  | 请求第几轮训练的数据，默认返回目前最新一轮的数据  
| metric | 否   | str  | 使用accuracy: 'accuracy', 使用loss: 'loss'，默认使用accuracy |