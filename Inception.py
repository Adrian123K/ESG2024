from tensorflow.keras import models, layers, applications
import tensorflow as tf

class Inception(layers.Layer): #inception 모듈
    def __init__(self, output_feature=32, mode='contract', **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.output_feature = output_feature #출력 데이터 채널 수
        self.mode = mode
        assert output_feature%4==0

    def build(self, input_shape): #사용할 레이어를 선언하는 메소드
        self.path1_conv11   = layers.Conv2D(self.output_feature//4,(1,1), padding='same', activation='relu' )

        self.path2_conv11   = layers.Conv2D(self.output_feature//4,(1,1), padding='same', activation='relu' )
        self.path2_conv33   = layers.Conv2D(self.output_feature//4,(3,3), padding='same', activation='relu' )

        self.path3_conv11   = layers.Conv2D(self.output_feature//4,(1,1), padding='same', activation='relu' )
        self.path3_conv33_1 = layers.Conv2D(self.output_feature//4,(3,3), padding='same', activation='relu' )
        self.path3_conv33_2 = layers.Conv2D(self.output_feature//4,(3,3), padding='same', activation='relu' )

        self.path4_conv33   = layers.Conv2D(self.output_feature//4,(3,3), padding='same', activation='relu' )
        self.path4_conv11   = layers.Conv2D(self.output_feature//4,(1,1), padding='same', activation='relu' )

        self.Batch          = layers.BatchNormalization()
        super(Inception, self).build(input_shape)

    def call(self, x): # 데이터 흐름(레이어 to 레이어)을 정의, 데이터는 call 메소드를 통해 연산
        path1 = self.path1_conv11(x)

        path2 = self.path2_conv11(x)
        path2 = self.path2_conv33(path2)

        path3 = self.path3_conv11(x)
        path3 = self.path3_conv33_1(path3)
        path3 = self.path3_conv33_2(path3)

        path4 = self.path4_conv33(x)
        path4 = self.path4_conv11(path4)

        together = layers.Concatenate()([path1, path2, path3, path4])

        return self.Batch(together)