from tensorflow.keras.layers import Input,Dense, Flatten, Conv2D,ReLU,BatchNormalization,Add
from keras.regularizers import l2

def conv_layer(input_block,filter_num,kernel_size,batch_norm = True,add_skip_con = False):
    x = Conv2D(filter_num,kernel_size,strides = (1,1),activation = 'linear',padding = 'same',data_format="channels_first",kernel_regularizer=l2(0.0001))(input_block)
    if batch_norm:
        x = BatchNormalization(axis = 1)(x)
    if add_skip_con:
        x = Add()([input_block,x])
    x = ReLU()(x)
    return x


def residual_layer(input_block,filter_num,kernel_size):

    x = conv_layer(input_block=input_block,
                    filter_num=filter_num,
                    kernel_size=kernel_size)
    x = conv_layer(input_block=x,
                    filter_num=filter_num,
                    kernel_size=kernel_size,
                    add_skip_con= True)
    return x

def input_conv_layer():
    main_input = Input(shape = (3,6,7), name = 'main_input')
    x = conv_layer(input_block = main_input,filter_num = 128 , kernel_size= (4,4))
    return main_input , x

def residual_tower (input_block,res_number = 5): 
    x = residual_layer(input_block=input_block,filter_num=128,kernel_size=(4,4))
    for num in range(res_number-1):
        x = residual_layer(input_block=x,filter_num=128,kernel_size=(4,4))
    return x

def policy_head (input_block):
    x = conv_layer(input_block=input_block,filter_num=2,kernel_size=(1,1))
    x = Flatten()(x)
    x = Dense(7,activation='softmax',name = 'policy_head')(x)
    return x

def value_head (input_block):
    x = conv_layer(input_block=input_block,filter_num=1,kernel_size=(1,1))
    x = Flatten()(x)
    x = Dense(96,activation='linear')(x)
    x = ReLU()(x)
    x = Dense(1,activation='tanh',name = 'value_head')(x)
    return x
