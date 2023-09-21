# KDD Cup 2022 - Baidu Spatial Dynamic Wind Power Forecasting
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/tanlongxing/kdd-cup-2022-wind-power-forecast)
[![arxiv](https://img.shields.io/badge/cs.ML-2307.09248-red?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2307.09248)

This is the 3rd place solution in Baidu KDD Cup 2022. The task is to predict the wind farm's future 48 hours active power for every 10 minutes.

<h1 align="center">
<img src="./data/user_data/model.png" width="700" align=center/>
</h1><br>


## Solution summary
- A single Transformer/ BERT model is made from [the tfts library](https://github.com/LongxingTan/Time-series-prediction). Follow its latest development [here](https://github.com/LongxingTan/Time-series-prediction)
- Using sliding window to generate more samples
- Only 2 raw features are used, wind speed and direction
- The daily fluctuation is added by post-processing to make the predicted result in line with daily periodicity


## How to use it

0. Prepare the tensorflow environment
```shell
pip install -r requirements.txt
```
1. Download the data from [Baidu AI studio](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction), and put it in `./data/raw`
2. Train the model, the file `result.zip` in `./weights/` can be used for submit. 
```shell
cd src/train
python nn_train.py
```


## Citation

If you find it useful in your research, please consider cite:

```
@article{tan2023application,
  title={Application of BERT in Wind Power Forecasting-Teletraan's Solution in Baidu KDD Cup 2022},
  author={Tan, Longxing and Yue, Hongying},
  journal={arXiv preprint arXiv:2307.09248},
  year={2023}
}
```


## Reference

```
[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
[2] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. 2021. Autoformer: De-composition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems 34 (2021), 22419–22430.
[3] JingboZhou,ShuangliLi,LiangHuang,HaoyiXiong,FanWang,TongXu,Hui Xiong, and Dejing Dou. 2020. Distance-aware molecule graph attention network for drug-target binding affinity prediction. arXiv preprint arXiv:2012.09624 (2020).
[4] HaoyiZhou,ShanghangZhang,JieqiPeng,ShuaiZhang,JianxinLi,HuiXiong, and Wancai Zhang. 2021. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35. 11106–11115.
```
