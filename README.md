# detect-facial-expressions

Python Application that can turn face expressions into emojis. 
For that I will train a Machine Learning Model with TensorFlow to detect face expressions like happy, sad, fearful and so on. 
Then we will map those expressions to Emojis in real time.

## How to run project

0. download training file
- download train data from [here](https://www.kaggle.com/datasets/msambare/fer2013?resource=download)
- download opencv from [here](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
- download image from [here](https://getavataaars.com/)
- put all the file into data folder

1. install dependency
```
    pip install -r requirements.txt
```

2. run model
```
    python3 train.py
```

3. run code
```
    streamlit run emoji2.py
```

## Reference

https://www.youtube.com/watch?v=nClG2ailhhk&t=368s