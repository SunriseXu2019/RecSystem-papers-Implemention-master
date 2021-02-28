
#Criteo Dataset

## Introduction：
Criteo是Kaggle上非常经典的[点击率预估比赛](https://www.kaggle.com/c/criteo-display-ad-challenge)，提供的数据集中训练集共4千多万行，测试集6百多万行，连续型特征共有13个，类别型特征共26个，样本按时间排序。


## Preprocess：
- 缺失值处理：

``` python
 df[sparse_features] = df[sparse_features].fillna('-1')
 df[dense_features] = df[dense_features].fillna(0)
```
- 类别特征编码和离散数据归一化

``` python
 for feat in sparse_features:
 	df[feat] = LabelEncoder().fit_transform(df[feat])
 df[dense_features] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[dense_features])
```
- 构造dense feature和sparse feature的字典，用于后续的embedding

``` python
   # 2.count #unique features for each sparse field and record dense feature field name
    dense_feature_columns = [{'name': feat} for feat in dense_features]
    sparse_feature_columns = [{'name': feat, 'feat_num': len(data_df[feat].unique())}
                              for feat in sparse_features]

    with open(os.path.join(data_path, 'dense.info'), 'wb') as f:
        pickle.dump(dense_feature_columns, f)
        f.close()

    with open(os.path.join(data_path, 'sparse.info'), 'wb') as f:
        pickle.dump(sparse_feature_columns, f)
        f.close()
```
- 建立lmdb索引，加快训练时数据的加载速度(对于4kw的数据量，整个txt文件从实验室服务器的ssd读到内存居然要花近30分钟，实在是太慢了，使用lmdb加速这个过程)

``` python
	# create lmdb dataset for criteo dataset
    with lmdb.open(os.path.join(cache_path, 'train.lmdb'), map_size=int(1e11)) as env:
        with env.begin(write=True) as txn:
            for index in tqdm(range(0, len(df))):
                line = np.array(df.loc[index, :])
                txn.put(struct.pack('>I', index), line.tobytes())
   print('success put train data in lmdb')
```









