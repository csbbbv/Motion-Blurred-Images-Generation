import random
import math,cmath
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.stats import poisson
import numpy as np
from numpy.core.defchararray import title
from numpy.core.fromnumeric import mean, size
# import sympy as sp

def tand(x):
    return math.tan(math.radians(x))

def sind(x):
    return math.sin(math.radians(x))

def cosd(x):
    return math.cos(math.radians(x))

# def triangle_fun(d):
#     d_new = []
#     for i in d:
#         d_new.append(max(1-abs(i),0))
#     return np.array(d_new)
# def triangle_fun_prod(d1,d2):
#     # d1,d2 = np.array(d1),np.array(d2)
#     d1,d2 = triangle_fun(d1),triangle_fun(d2)
#     return d1*d2#np.dot(d1,d2)

def shift(lst, k):
  x = lst[:k]
  x.reverse()
  y = lst[k:]
  y.reverse()
  r = x+y
  return list(reversed(r))



def cicleshift(lst,y,x):
    y = -y
    x = -x
    new = []
    lst = lst.tolist()
    for i in range(len(lst)):
        new.append(shift(lst[i],x))
    return np.array(shift(new,y))

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def createTrajectory(TrajSize, anxiety, numT, MaxTotalLength, do_show):
    '''
    % input description
    % TrajSize                   size (in pixels) of the square support of the Trajectory curve
    % anxiety                    parameter determining the amount of shake (in the range [0,1] 0 corresponds to rectilinear
    %                                 trajectories). This term scales the random vector that is added at each sample.
    % numT                      number of samples where the Trajectory is sampled
    % MaxTotalLength    maximum length of the trajectory computed as the sum of all distanced between consecuive points
    % do_show                show a figure illustrating the trajectory
    %
    % output description
    % TrajCurve                                  ouput structure having the following fields
    %   TrajCurve.x                             complex-valued vector. Each point determines a position in the complex plane to be considered as
    %                                                           the image plane
    % TrajCurve.TotLenght                length of the TrajCurve (measured as the sum of absolute distance between consecutive points )
    % TrajCurve.Anxiety                     input parameter
    % TrajCurve.MaxTotalLength      input parameter
    % TrajCurve.nAbruptShakes       number of abrupt shakes occurred here'''
    do_compute_curvature = 0
    TotCurvature = 0
    TotLength = 0
    abruptShakesCounter = 0


    centripetal = 0.7 * random.random()
    gaussianTerm =10 * random.random()
    freqBigShakes = 0.2 * random.random()
    init_angle = 360 * random.random()

    v0 = cosd(init_angle) + sind(init_angle)*1j
    v = v0 * MaxTotalLength/(numT-1)

    if anxiety > 0:
        v = v0 * anxiety

    x = np.zeros(numT)
    x = np.array(x,dtype=complex)

    for t in range(numT-1):
        if random.random() < freqBigShakes * anxiety:
            nextDirection = 2 * v * (cmath.exp(1j*(math.pi+(random.random()-0.5))))
            abruptShakesCounter = abruptShakesCounter + 1
        else:
            nextDirection=0
        dv = nextDirection + anxiety * (gaussianTerm * (random.gauss(0,1) + 1j * random.gauss(0,1)) - centripetal * x[t]) * (MaxTotalLength / (numT - 1))
        v=v+dv
        v = (v / abs(v)) * MaxTotalLength / (numT - 1)
        x[t + 1] = x[t] + v
        TotLength=TotLength+abs(x[t+1]-x[t])
        # if do_compute_curvature:
        #     if t >0:
        #         TotCurvature = TotCurvature+
    imag_x = []
    real_x = []
    for i in x:
        imag_x.append(i.imag)
        real_x.append(i.real)
    x = x - 1j*min(imag_x)-min(real_x)

    x = x - 1j * (x[0].imag % 1) - (x[0].real % 1) + 1 + 1j
    imag_x = []
    real_x = []
    for i in x:
        imag_x.append(i.imag)
        real_x.append(i.real)
    x = x + 1j * math.ceil((TrajSize - max(imag_x))/2) + math.ceil((TrajSize - max(real_x))/2)
    imag_x = []
    real_x = []
    for i in x:
        imag_x.append(i.imag)
        real_x.append(i.real)

    if do_show:
        plt.figure(455)
        p1, = plt.plot(real_x,imag_x)
        # p1, = plt.plot(x)
        p2, = plt.plot(x[0].real,x[0].imag,'rx')
        p3, = plt.plot(x[-1].real,x[-1].imag,'ro')
        plt.axis([0,TrajSize,0,TrajSize])
        plt.legend([p1,p2,p3],('Traj Curve', 'init' , 'end'))
        plt.title(['anxiety:' , str(anxiety) , ' number of abrupt shakes: ', str(abruptShakesCounter)])
        plt.show()
    
    TrajCurve = {}
    TrajCurve['x'] = x
    TrajCurve['TotLenght'] = TotLength
    TrajCurve['TotCurvature'] = TotCurvature
    TrajCurve['Anxiety'] = anxiety
    TrajCurve['MaxTotalLength'] = MaxTotalLength
    TrajCurve['nAbruptShakes'] = abruptShakesCounter
    return TrajCurve

    

def createPSFs(TrajCurve , PSFsize , T , do_show , do_center):
    '''
    % input description
    % TrajCurve                 Motion Blur trajectory cuve, provided by createTrajectory function
    % PSFsize                     Size of the PFS where the TrajCurve is sampled
    % T                                Vector of exposure times: for each of them a PSF will be generated
    % do_show
    % do_center
    %
    % output description
    % PSFS                       cell array containing PSFS sampling TrajCurve for each exposure time  in T
    %                                   numel(PSFS) = length(T).
    '''

    PSFsize=[PSFsize,PSFsize]

    PSFnumber = len(T)
    numt = len(TrajCurve['x'])
    x= TrajCurve['x']

    if do_center:
        x = x-mean(x)+(PSFsize[1]+1j*PSFsize[0]+1+1j)/2
    
    # generate le PSFS
    PSFS = [[] for i in range(PSFnumber)]

    PSF = np.zeros((PSFsize[0],PSFsize[1]))

    triangle_fun = lambda d : max(0,(1-abs(d)))
    triangle_fun_prod = lambda d1,d2 :triangle_fun(d1)*triangle_fun(d2)
    for jj in range(len(T)):
        if jj == 0:
            prevT = 0
        else:
            prevT = T[jj - 2]
        for t in range(len(x)):
            if (T[jj]*numt >= t) and (prevT * numt < t-1):
                t_proportion = 1
            elif (T[jj]*numt >= t-1) and (prevT * numt < t-1) :
                t_proportion = (T[jj] * numt)-(t-1)
            elif (T[jj]*numt >= t) and (prevT * numt < t):
                t_proportion = t - (prevT * numt)
            elif (T[jj]*numt >= t - 1 ) and (prevT * numt < t):
                t_proportion = (T[jj] - prevT) * numt
            else:
                t_proportion = 0

            m2 = min(PSFsize[1] - 1 , max(1,math.floor(x[t].real)))
            M2 = m2 + 1
            m1 = min(PSFsize[0]-1 , max(1,math.floor(x[t].imag)))
            M1 = m1 + 1

            PSF[m1][m2] = PSF[m1][m2] + t_proportion * triangle_fun_prod(x[t].real - m2 , x[t].imag - m1)
            PSF[m1][M2] = PSF[m1][M2] + t_proportion * triangle_fun_prod(x[t].real - M2 , x[t].imag - m1)
            PSF[M1][m2] = PSF[M1][m2] + t_proportion * triangle_fun_prod(x[t].real - m2 , x[t].imag - M1)
            PSF[M1][M2] = PSF[M1][M2] + t_proportion * triangle_fun_prod(x[t].real - M2 , x[t].imag - M1)

        PSFS[jj] = PSF/len(x)
    if do_show:
        C = np.array([])
        D = np.array([])
        for jj in range(len(T)):
            if jj == 0:
                C = PSFS[jj]
                # D = PSFS[jj]/max(PSFS[jj][:])
                D = PSFS[jj]/max(map(max,PSFS[jj]))
            else:
                C = np.hstack((C,PSFS[jj]))
                D = np.hstack((D,PSFS[jj]/max(map(max,PSFS[jj]))))

                # C.append(PSFS[jj])
                # D.append(PSFS[jj]/max(PSFS[jj][:]))
        plt.figure(456)
        plt.subplot(1,2,1)
        plt.imshow(C)
        plt.title("all PSF normalized w.r.t. the same maximum")
        plt.subplot(1,2,2)
        plt.imshow(D)
        plt.title("each PSFs normalized w.r.t. its own maximum")
        plt.colormaps()
        plt.hot()
        plt.show()
    
    return PSFS

def createBlurredRaw(y, psf, lambda_, sigma_gauss):
    '''
    %  [Raw,V]= CreateBlurredRaw(y, psf, lambda, sigma_gauss, init)
    % output description
    % Raw           noisy blurred observation
    % V             Fourier Transform of PSF (sized as Raw)
    % 
    % input description
    % y                original image
    % psf              psf in space domain
    % lambda           Poisson noise parameter
    % sigma_gauss      Gaussian noise parameter
    % init             (optional) initialization parameter for Poissonian and Gaussian noise'''
    random.seed(1)
    y = y * lambda_
    yN,xN = y.shape
    ghy,ghx = psf.shape
    big_v = np.zeros((yN,xN))
    for i in range(ghy):
        for j in range(ghx):
            big_v[i][j] = psf[i][j]
    big_v = cicleshift(big_v,-round((ghy-1)/2),-round((ghx-1)/2))
    V = np.fft.fft2(big_v)
    y_blur = np.fft.ifft2(V * np.fft.fft2(y)).real


    y_blur_new = np.zeros((len(y_blur),len(y_blur[0])))
    for i in range(len(y_blur)):
        for j in range(len(y_blur[0])):
            y_blur_new[i][j] = y_blur[i][j]>0
    Raw = np.random.poisson(np.array(y_blur)*np.array(y_blur_new))
    Raw = Raw + sigma_gauss*np.random.randn(Raw.shape[0],Raw.shape[1])

    return Raw
    
def demo(img):
    idx = random.randint(0, 3)
    do_show = 1
    PSFsize = 64
    anxiety = 0.005
    numT = 2000 
    MaxTotalLength = 64
    T = [0.0625 , 0.25 , 0.5, 1]
    do_centerAndScale = 0
    lambda_ = 2048
    sigmaGauss = 0.05

    # img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    # y =im2double(img)

    # TrajCurve = createTrajectory(PSFsize, anxiety, numT, MaxTotalLength, do_show)
    # PSFs = createPSFs(TrajCurve, PSFsize,  T , do_show , do_centerAndScale)
    
    TrajCurve = createTrajectory(PSFsize,anxiety,numT,MaxTotalLength,do_show)

    PSFs = createPSFs(TrajCurve,PSFsize,T,do_show,do_centerAndScale)

    # zeroCol = []
    # paddedImage = [zeroCol]
    cnt = 0
    for ii in range(len(PSFs)):
        Raw  = createBlurredRaw(img,PSFs[ii],lambda_,sigmaGauss)
        if cnt == idx:
            return Raw
        cnt += 1

if __name__ == '__main__':
    
    do_show = 1
    PSFsize = 64
    anxiety = 0.005
    numT = 2000 
    MaxTotalLength = 64
    T = [0.0625 , 0.25 , 0.5, 1]
    do_centerAndScale = 0
    lambda_ = 2048
    sigmaGauss = 0.05

    img = cv.imread('img\\3d\\20210108\\72.jpg',cv.IMREAD_GRAYSCALE)
    y =im2double(img)

    # TrajCurve = createTrajectory(PSFsize, anxiety, numT, MaxTotalLength, do_show)
    # PSFs = createPSFs(TrajCurve, PSFsize,  T , do_show , do_centerAndScale)
    
    TrajCurve = createTrajectory(64,0.005,2000,64,1)

    PSFs = createPSFs(TrajCurve,64,T,do_show,do_centerAndScale)

    zeroCol = []
    paddedImage = [zeroCol]
    for ii in range(len(PSFs)):
        Raw  = createBlurredRaw(y,PSFs[ii],lambda_,sigmaGauss)
        plt.figure()
        plt.imshow(Raw,cmap='gray')
        imTemp = Raw / max(map(max,Raw))
        imTemp[:]
        plt.title(['image having exposure time ', str(T[ii])])
        plt.show()
    # path = 'img\\3d\\20210108\\72.jpg'
    # img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    # img =im2double(img)
    # Raw  = demo(img)
    # plt.figure()
    # plt.imshow(Raw,cmap='gray')
    # imTemp = Raw / max(map(max,Raw))
    # imTemp[:]
    # plt.title(['image having exposure time ', str(T[ii])])
    plt.show()






