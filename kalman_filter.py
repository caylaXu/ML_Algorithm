#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 17-6-15 下午8:23
# @Author  : CaylaXu
# @File    : kalman_filter.py
# @Description: 简单的卡尔曼滤波
# current estimation = Gain * Measured + (1 - Gain) * previousestimation
# 当前估计=增益*度量+(1 -增益)*预估

import pylab
class KalmanFilter:
    def __init__(self):
        self.x = 0
        self.p = 0.1
        self.A = 1
        self.H = 1
        self.q = 10e-6 #10e-6;  /* predict noise convariance */
        self.r = 10e-5 #10e-5;  /* measure error convariance */

    def measurement_update(self,measurement):
        #Predict
        self.x = self.A * self.x
        self.p = self.A * self.A * self.p + self.q;  # p(n|n-1)=A^2*p(n-1|n-1)+q
        #Measurement
        gain = self.p * self.H / (self.p * self.H * self.H + self.r);
        self.x = self.x + gain * (measurement - self.H * self.x);
        self.p = (1 - gain * self.H) * self.p;
        return self.x,self.p,gain

if __name__ == '__main__':
    k = KalmanFilter()
    measurements = [0.0,0.0,0.0,0.27,0.28,0.29,0.3,0.31,0.3,0.29,0.27,0.27,0.28,0.29,0.3,
                    0.31,0.3,0.29,0.27,0.27,0.28,0.29,0.3,0.31,0.3,0.29,0.27,0.27,0.28,0.29,
                    0.3,0.31,0.3,0.29,0.27,0.27,0.28,0.29,0.3,0.31,0.3,0.29,0.27,]
    estimate = [0]
    noise = [1]
    for val in measurements:
        new_est, new_noise, new_gain = k.measurement_update(val)
        estimate.append(new_est)
        noise.append(new_noise)
    pylab.plot(estimate)
    pylab.plot(measurements)
    pylab.show()
