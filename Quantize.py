# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:50:44 2018

@author: amax
"""

import tensorflow as tf
from utils.argtools import graph_memoized


@graph_memoized
def get_fw_fa(bitsW, bitsA,bitsG,bitsE1,bitsE2,bitsBN_G,bitsBN_B,bitsBN_mean,bitsBN_var,bitsBN_x,bitsLR,bits_gBN):
    
    @tf.custom_gradient
    def Q(x, bits):
        x=tf.cast(x,tf.float32)
        bits=tf.cast(bits,tf.float32)
        n=tf.pow(2.0,bits-1)
        y=tf.round(x*n)/n
        def grad(dy):
            return dy
        return y,grad
    
    def clip(x,bits):
        if bits>=32:
            delta=0.0
        else:
            x=tf.cast(x,tf.float32)
            bits=tf.cast(bits,tf.float32)
            delta=1./tf.pow(2.0,bits-1)
        MAX=+1-delta
        MIN=-1+delta
        x=tf.clip_by_value(x,MIN,MAX,name='saturate')
        return x
    
    def Shift(x):
        return 2 ** tf.round(tf.log(x) / tf.log(2.0))
    def S(bits):
        return 2.0 ** (bits - 1)
        
    @tf.custom_gradient
    def fw(x):
        def grad(dy):
            return dy            
        if bitsW >=32:
            return x,grad
        else:
            return clip(Q(x,bitsW),bitsW),grad
           
    @tf.custom_gradient
    def fa(x):
        def grad(dy):
            return dy
        if bitsA>=32:
            return x,grad
        else:
            return Q(x,bitsA),grad
    
    @tf.custom_gradient
    def fbn_G(x):
        def grad(dy):
            return dy
        if bitsBN_G>=32:
            return x,grad
        else:
            return Q(x,bitsBN_G),grad
    
    @tf.custom_gradient
    def fbn_B(x):
        def grad(dy):
            return dy
        if bitsBN_B>=32:
            return x,grad
        else:
            return Q(x,bitsBN_B),grad
    
    @tf.custom_gradient
    def fbn_mean(x):
        def grad(dy):
            return dy
        if bitsBN_mean>=32:
            return x,grad
        else:
            return Q(x,bitsBN_mean),grad
            
    @tf.custom_gradient
    def fbn_var(x):
        def grad(dy):
            return dy
        if bitsBN_var>=32:
            return x,grad
        else:
            return Q(x,bitsBN_var),grad
    

    
    @tf.custom_gradient
    def fbn_x(x):
        def grad(dy):
            return dy
        if bitsBN_x>=32:
            return x,grad
        else:
            return Q(x,bitsBN_x),grad
    

    @tf.custom_gradient
    def fg(x,lr,g_scale):
        def grad(dy):
            return dy
        if bitsG>=32:
            return lr*x,grad
        else:
            bitsR = 32
            xmax = tf.reduce_max(tf.abs(x))
            x = x / Shift(xmax)
            


            #LR = 128.0
            LR = g_scale
            norm = Q(LR * x, bitsR)

            norm_sign = tf.sign(norm)
            norm_abs = tf.abs(norm)
            norm_int = tf.floor(norm_abs)
            norm_float = norm_abs - norm_int
            rand_float = tf.random_uniform(x.get_shape(), 0, 1)
            norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )
            norm = tf.clip_by_value(norm,-LR+1,LR-1)
            G=Q(lr*norm / (128*S(bitsG)),15)
            return G,grad
    

     
    @tf.custom_gradient
    def fgBN(x,lr):
       def grad(dy):
           return dy
       if bits_gBN>=32:
           return lr*x,grad
       else:
           x=Q(x,bits_gBN)
           return Q(lr*x,bits_gBN),grad  
            
            
            
    
    
    @tf.custom_gradient
    def fe1(x):
        def grad(dy):
            if bitsE1>=32:
                return dy
            else:
                dymax = tf.reduce_max(tf.abs(dy))
                dymax_shift = Shift(dymax)
                dy_q=dymax_shift*tf.clip_by_value(Q(dy /dymax_shift, bitsE1),-1,1)
                print(dy_q.name)
                return  dy_q              
                                 
        return x,grad
    
    
    
    @tf.custom_gradient
    def fe2(x):
        def grad(dy):
            if bitsE2>=32:
                return dy
            else:
              dymax = tf.reduce_max(tf.abs(dy))
              dymax_shift = Shift(dymax)
                
              dymax_s = dymax_shift/(2**(bitsE2-1))                
              dy_s = dy / dymax_s
              sign = tf.sign(dy_s)
              dy_s = tf.abs(dy_s)

              zero = tf.zeros_like(dy_s)
              dy_s1 = tf.where(dy_s<1,x=dy_s,y=zero)
              dy_s2 = tf.where(dy_s>=1,x=dy_s,y=zero)

              dy_s1 = sign*tf.clip_by_value(Q(dy_s1,bitsE2),-1,1)
              dy_s2 = sign*tf.clip_by_value(tf.round(dy_s2),-2**(bitsE2-1)+1,2**(bitsE2-1)-1)

              dy_s = dy_s1 + dy_s2
              
              E2=dymax_s*dy_s
              return E2

        return x,grad
    

    @tf.custom_gradient
    def flr(x):
        def grad(dy):
            return dy            
        if bitsLR >=32:
            return x,grad
        else:
            return clip(Q(x,bitsLR),bitsLR),grad   


    def fBits(x,bits=32):           
        if  bits >=32:
           return x
        else:
           return Q(x,bits)    
    
    return fw,fa,fg,fe1,fe2,fbn_G,fbn_B,fbn_mean,fbn_var,fbn_x,flr,fgBN,fBits



bitsW=8
bitsA=8
bitsG=8
bitsE1=8
bitsE2=8
bitsBN_G=8
bitsBN_B=8
bitsBN_mean=16
bitsBN_var=16
#bitsBN_x=16
bitsBN_x=8
bitsLR=10
bits_gBN=15



fw,fa,fg,fe1,fe2,fbn_G,fbn_B,fbn_mean,fbn_var,fbn_x,flr,fgBN,fBits=\
  get_fw_fa(bitsW=bitsW, bitsA=bitsA,bitsG=bitsG,bitsE1=bitsE1,bitsE2=bitsE2,bitsBN_G=bitsBN_G,bitsBN_B=bitsBN_B,bitsBN_mean=bitsBN_mean,bitsBN_var=bitsBN_var,bitsBN_x=bitsBN_x,bitsLR=bitsLR,bits_gBN=bits_gBN)


         
        
    
         
