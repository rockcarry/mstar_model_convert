import numpy as np
import doctest
import math

class anchor:
    @staticmethod
    def zeros(shape):
        '''
        doing nothing just return the input
        :param input_tensor:
        :return:
        '''
        return np.zeros(shape)

    @staticmethod
    def ones(shape):
        '''

        :return:
        '''
        return np.ones(shape)

    @staticmethod
    def ns(shape,n):
        '''

        :return:
        '''
        return np.ones(shape)*n

    @staticmethod
    def full(shape,fill_value=0):
        return np.full(shape,fill_value)

    @staticmethod
    def get_anchors_num(h,w):
        sgs_feature_map_sizes = []
        conv1_h = np.floor((h + 2.0*3.0-(7.0-1.0)-1.0)/4.0 + 1.0)
        conv1_w = np.floor((w + 2.0 * 3.0-(7.0 -1.0)-1.0)/4.0 + 1.0)
        pool1_h = np.ceil((conv1_h + 2.0* 0-(3.0- 1.0)-1.0) / 2.0+ 1)
        pool1_w = np.ceil((conv1_w + 2.0* 0-(3-1)-1) / 2.0+ 1)
        conv2_h = np.floor((pool1_h + 2.0* 2.0- (5.0- 1)-1) / 2.0+ 1)
        conv2_w = np.floor((pool1_w + 2.0* 2.0- (5.0- 1)-1) / 2.0+ 1)
        pool2_h = np.ceil((conv2_h + 2.0* 0-(3.0- 1)-1) / 2.0+ 1)
        pool2_w = np.ceil((conv2_w + 2.0* 0-(3.0- 1)-1) / 2.0+ 1)
        sgs_feature_map_sizes.append(pool2_h)
        sgs_feature_map_sizes.append(pool2_w)
        conv3_1_h = np.floor((pool2_h + 2.0* 0-(1-1)-1) / 1 + 1)
        conv3_1_w = np.floor((pool2_w + 2.0* 0-(1-1)-1) / 1 + 1)
        conv3_2_h = np.floor((conv3_1_h + 2.0* 1-(3-1)-1) / 2.0+ 1)
        conv3_2_w = np.floor((conv3_1_w + 2.0* 1-(3-1)-1) / 2.0+ 1)
        sgs_feature_map_sizes.append(conv3_2_h)
        sgs_feature_map_sizes.append(conv3_2_w)
        conv4_1_h = np.floor((conv3_2_h + 2.0* 0-(1.0- 1.0)-1.0) / 1.0+ 1.0)
        conv4_1_w = np.floor((conv3_2_w + 2.0* 0-(1-1)-1) / 1 + 1)
        conv4_2_h = np.floor((conv4_1_h + 2.0* 1-(3-1)-1) / 2.0+ 1)
        conv4_2_w = np.floor((conv4_1_w + 2.0* 1-(3-1)-1) / 2.0+ 1)
        sgs_feature_map_sizes.append(conv4_2_h)
        sgs_feature_map_sizes.append(conv4_2_w)

        return sgs_feature_map_sizes



    # weight*x+bias , x=index*step index in range(period) repeat times
    @staticmethod
    def index_mod_linear(step,weight,bias,period,repeat1,repeat2=1):
        temp= np.fromfunction(lambda x: weight*(x%period*step) +bias, (period*repeat1,),dtype=int)
        return np.tile(temp,repeat2)

    @staticmethod
    def index_div_linear(step,weight,bias,period,repeat1,repeat2=1):
        temp = np.fromfunction(lambda x: weight*(x//period*step) +bias, (period*repeat1,),dtype=int)
        return np.tile(temp,repeat2)


    @staticmethod
    def fromfunc_mod_linear(func,step,bias,period,repeat,repeat2=1):
        temp = np.fromfunction(lambda x:  func(x%period*step) +bias, (period*repeat,),dtype=int)
        return np.tile(temp,repeat2)

    @staticmethod
    def fromfunc_div_linear(func,step,bias,period,repeat, repeat2):
        temp = np.fromfunction(lambda x: func(x//period*step) +bias, (period*repeat,),dtype=int)
        return np.tile(temp,repeat2)

if __name__=='__main__':
    import doctest
    doctest.testmod(verbose=True)



