from tensorflow.keras import models, layers, applications
import tensorflow as tf

class Expand(layers.Layer): #expanding path 모듈
    def __init__(self, output_feature=32, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.output_feature = output_feature #출력 데이터 채널 수

    def build(self, input_shape): #사용할 레이어를 선언하는 메소드
        self.Conv1  = layers.Conv2D(self.output_feature,(3,3), padding='same', activation='relu' )
        self.Conv2  = layers.Conv2D(self.output_feature,(3,3), padding='same', activation='relu' )
        super(Expand, self).build(input_shape)

    def call(self, X): # 데이터 흐름(레이어 to 레이어)을 정의, 데이터는 call 메소드를 통해 연산
        x = layers.Concatenate()(X)
        x1 = self.Conv1(x)
        x1 = self.Conv2(x1)
        return x1

