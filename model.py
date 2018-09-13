from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam, SGD
from  tensorflow.python.keras import backend as K

class Model_OCR:
    def __init__(self):
        self.feature_extractor = keras.applications.MobileNet(weights='imagenet', include_top=False)
        self.rnn = CuDNNGRU
        self.height=48
        self.rnnunit = 256
        self.nclass = 11
        self.build_model()

    def build_model(self):
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            y_pred = y_pred[:, 2:, :]
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        inputs = Input(shape=(self.height,None,3),name='the_input')
        m = self.feature_extractor(inputs)
        for layer in self.feature_extractor.layers: layer.trainable=False
        #2. bi-lstm layers
        m = Permute((2,1,3),name='permute')(m)
        m = TimeDistributed(Flatten(),name='timedistrib')(m)
        m = Bidirectional(self.rnn(self.rnnunit,return_sequences=True),name='blstm1')(m)
        m = Dense(self.rnnunit,name='blstm1_out',activation='linear')(m)
        m = Bidirectional(self.rnn(self.rnnunit,return_sequences=True),name='blstm2')(m)
        y_pred = Dense(self.nclass,name='blstm2_out',activation='softmax')(m)

        basemodel = Model(inputs=inputs,outputs=y_pred)
        #3. CTC loss compute
        labels = Input(name='the_labels', shape=[None,], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        self.model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
        self.test_func = K.function([inputs], [y_pred])
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

if __name__ == '__main__':
    model_ocr = Model_OCR()