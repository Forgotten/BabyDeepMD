import numpy as np

def potential_Per(x,y, mu,L):
    return np.exp( -mu*np.minimum(np.abs(y - x), L - np.abs(y - x)) )

def forces_Per(x,y, mu,L):
    return -mu*np.sign(y - x)*np.exp( -mu*np.minimum( np.abs(y - x),L-np.abs(y - x)) )

def gen_data_Per_Mixed(Ncells, Np, mu1, mu2, Nsamples, minDelta = 0.0, Lcell = 0.0 , weight1=0.5 , weight2=0.5):
    
    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))
    
    if Lcell == 0.0 :
        sizeCell = 1/Ncells
    else :
        sizeCell = Lcell 
    L = sizeCell*Ncells
    
    for i in range(Nsamples):
        midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsExt = np.concatenate([points - L, points, points + L])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:            
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)
            pointsExt = np.concatenate([points - L, points, points + L])
            
        pointsArray[i, :] = points.T

        R1 = potential_Per(points,points.T, mu1,L)
        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)
        R2 = potential_Per(points,points.T, mu2,L)
        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)
        potentialArray[i,:] = weight1*potTotal1+weight2*potTotal2

        F1 = forces_Per(points,points.T, mu1,L)
        Forces1 = np.sum(F1, axis = 1)
        F2 = forces_Per(points,points.T, mu2,L)
        Forces2 = np.sum(F2, axis = 1)
        forcesArray[i,:] = weight1*Forces1.T + weight2*Forces2.T

    return pointsArray, potentialArray, forcesArray

def gen_data_Per(Ncells, Np, mu, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))
    
    if Lcell == 0.0 :
        sizeCell = 1/Ncells
    else :
        sizeCell = Lcell 
    L = sizeCell*Ncells
    
    for i in range(Nsamples):
        midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsExt = np.concatenate([points - L, points, points + L])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:            
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)
            pointsExt = np.concatenate([points - L, points, points + L])
            
        pointsArray[i, :] = points.T

        R = potential_Per(points,points.T, mu,L)

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = forces_Per(points,points.T, mu,L)

        Forces = np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray

def potential(x,y, mu):
    return -np.exp(-mu*np.abs(y - x))

def forces(x,y, mu):
    return -mu*np.sign(y - x)*np.exp(-mu*np.abs(y - x))

def gen_data(Ncells, Np, mu, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    for i in range(Nsamples):
        if Lcell == 0.0 :
            sizeCell = 1/Ncells
        else :
            sizeCell = Lcell

        midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        # we want to check that the points are not too close
        while np.min(points[1:] - points[0:-1]) < minDelta:
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsArray[i, :] = points.T

        R = potential(points,points.T, mu)

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = forces(points,points.T, mu)

        Forces = np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray


def genDataYukawaPerMixed(Ncells, Np, sigma1, sigma2, Nsamples, minDelta = 0.0, Lcell = 0.0,weight1=0.5,weight2=0.5):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    if Lcell == 0.0 :
        sizeCell = 1/Ncells #calculate a density
    else :
        sizeCell = Lcell

#    NpointsPerCell = 1000
#    Nx = Ncells*NpointsPerCell + 1
    NpointsPerCell = int(1000*sizeCell) ###1000 points in an interval whose length is 1.0
#    Nx = 10001
    Nx = Ncells*NpointsPerCell + 1
#    NpointsPerCell = (Nx - 1)//Ncells
    Ls = Ncells*sizeCell
    print('here I make an adjustment')
    xGrid, pot1, dpotdx1 = computeDerPotPer(Nx, sigma1, Ls)
    xGrid, pot2, dpotdx2 = computeDerPotPer(Nx, sigma2, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)

    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

        pointsArray[i, :] = points.T

        R1 = pot1[idxPointCell - idxPointCell.T]##matrix interaction
        RR1 = np.triu(R1, 1) ###consider interaction effect
        potTotal1 = np.sum(RR1)
        R2 = pot2[idxPointCell - idxPointCell.T]##matrix interaction
        RR2 = np.triu(R2, 1) ###consider interaction effect
        potTotal2 = np.sum(RR2)
        potentialArray[i,:] = weight1*potTotal1 + weight2*potTotal2

        F1 = dpotdx1[idxPointCell - idxPointCell.T]
        F1 = np.triu(F1,1) + np.tril(F1,-1)
        Forces1 = -np.sum(F1, axis = 1)
        F2 = dpotdx2[idxPointCell - idxPointCell.T]
        F2 = np.triu(F2,1) + np.tril(F2,-1)
        Forces2 = -np.sum(F2, axis = 1)
        forcesArray[i,:] = weight1*Forces1.T + weight2*Forces2.T

    return pointsArray, potentialArray, forcesArray

def potentialGaussian(x, y, sigma):
    return -np.exp(-sigma*np.square(y - x))


def forcesGaussian(x, y, sigma):
    return -sigma*2*(y - x)*np.exp(-sigma*np.square(y - x))


def gen_dataGaussian(Ncells, Np, sigma, Nsamples, minDelta = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    for i in range(Nsamples):
        sizeCell = 1/Ncells
        midPoints = np.linspace(sizeCell/2.0,1-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        # we want to check that the points are not too close
        while np.min(points[1:] - points[0:-1]) < minDelta:
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsArray[i, :] = points.T

        R = potentialGaussian(points,points.T,sigma)

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = forcesGaussian(points,points.T,sigma)

        Forces = np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray



def potentialYukawa(x, y, mu):
    return -np.exp(-mu*np.abs(y - x))/np.abs(y - x)


def forcesYukawa(x, y, mu):
    return     mu*np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.abs(y - x) + np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.square(np.abs(y - x))


def genDataYukawa(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    for i in range(Nsamples):
        if Lcell == 0.0 :
            sizeCell = 1/Ncells
        else :
            sizeCell = Lcell

        midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

        points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
        points = np.sort(points.reshape((-1,1)), axis = 0)

        # we want to check that the points are not too close
        while np.min(points[1:] - points[0:-1]) < minDelta:
            points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
            points = np.sort(points.reshape((-1,1)), axis = 0)

        pointsArray[i, :] = points.T

        R = potentialYukawa(points,points.T,sigma)

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = forcesYukawa(points,points.T,sigma)
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray


def genDataYukawaPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    if Lcell == 0.0 :
        sizeCell = 1/Ncells #calculate a density
    else :
        sizeCell = Lcell

    NpointsPerCell = int(1000*sizeCell)
    Nx = Ncells*NpointsPerCell + 1    
    Ls = Ncells*sizeCell
    print('here I make an adjustment')
    xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)

    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

        pointsArray[i, :] = points.T

        R = pot[idxPointCell - idxPointCell.T]##matrix interaction

        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = dpotdx[idxPointCell - idxPointCell.T]
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = -np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray


def genDataYukawaPerlarge(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    if Lcell == 0.0 :
        sizeCell = 1/Ncells #calculate a density
    else :
        sizeCell = Lcell

    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell

    xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

    idxCell = np.linspace(100,NpointsPerCell-100-1, NpointsPerCell-200).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)

    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell]



        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

        pointsArray[i, :] = points.T

        R = pot[idxPointCell - idxPointCell.T]##matrix interaction

        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = dpotdx[idxPointCell - idxPointCell.T]
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = -np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray


def genDataYukawaPerIndex(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    idxArray = np.zeros((Nsamples, Ncells, Np)).astype(int64)
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))


    if Lcell == 0.0 :
        sizeCell = 1/Ncells #calculate a density
    else :
        sizeCell = Lcell

    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell

    xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int64)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
#    idxArray = np.zeros((Nsamples, Ncells,Np))
    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell1]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell1]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

#        idxArray[i,:,:] = idxPointCell
        pointsArray[i, :] = points.T
        idxArray[i,:,:] = idxPointCell

        R = pot[idxPointCell1 - idxPointCell1.T]##matrix interaction

        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = dpotdx[idxPointCell1 - idxPointCell1.T]
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = -np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T
        
#    idxArray = idxArray.astype(int)
    return idxArray, pointsArray, potentialArray, forcesArray


#def computeEFfromIndex(idxArray, Ncells, mu, sizeCell, xCenter = 0, nPointSmear = 10):
def computeEFfromIndex(idxArray, Ncells, mu, sizeCell, xCenter = 0, nPointSmear = 10):
    ######idxArray (Nsamples , Ncells , Np)
    ######for later index, we need to reduce them into the domain (shift)
    Nx = Ncells*1000 + 1
    Ls = Ncells*sizeCell
    xGrid = np.linspace(0, Ls, Nx+1)[:-1] ##delete the last one
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls ##frequency domain

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))
    ##yukuwa potential V = F(mult)

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = nPointSmear*Ls/Nx #standard deviation for guass, use guass to approach dirac

    x = gaussian(xGrid, xCenter, tau) + \
        gaussian(xGrid - Ls, xCenter, tau) +\
        gaussian(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x)) #FFT of delta function, shift is just for convenience

    yFFT = xFFT*mult ### in frequency domain, multiply a certain factor
                     ###convolution
    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT))) ## y = dirac convolution V

    dydxFFT = 1.j*kGrid*yFFT #1.j=1j 1st derivative
    dydx = np.real(np.fft.ifft(np.fft.ifftshift(dydxFFT)))


    Nsamples = idxArray.shape[0]
    Np = idxArray.shape[2]
#    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))  
    pointsArray = np.zeros((Nsamples,Np*Ncells))
    
    for j in range(Nsamples):
        idxPointCell = idxArray[j,:,:]
#        idxPointCell = idxArray[j,:,:]
        idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
        points = xGrid[idxPointCell1]
        pointsArray[j, :] = points.T
        
        R = y[idxPointCell1 - idxPointCell1.T]##matrix interaction
        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)
        potentialArray[j,:] = potTotal
        F = dydx[idxPointCell1 - idxPointCell1.T]
        F = np.triu(F,1) + np.tril(F,-1)
        Forces = -np.sum(F, axis = 1)
        forcesArray[j,:] = Forces.T
        
    return pointsArray, potentialArray, forcesArray

def gaussian(x, xCenter, tau):
    return (1/np.sqrt(2*np.pi*tau**2))*\
           np.exp( -0.5*np.square(x - xCenter)/tau**2 )

def gaussianNUFFT(x, xCenter, tau):
    return np.exp(-np.square(x - xCenter)/(4*tau) )

def gaussianDeconv(k, tau):
    return np.sqrt(np.pi/tau)*np.exp( np.square(k)*tau)

def computeDerPotPer(Nx, mu, Ls, xCenter = 0, nPointSmear = 10):

    xGrid = np.linspace(0, Ls, Nx+1)[:-1] ##delete the last one
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls ##frequency domain

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))
    ##yukuwa potential V = F(mult)

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = nPointSmear*Ls/Nx #standard deviation for guass, use guass to approach dirac

    x = gaussian(xGrid, xCenter, tau) + \
        gaussian(xGrid - Ls, xCenter, tau) +\
        gaussian(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x)) #FFT of delta function, shift is just for convenience

    yFFT = xFFT*mult ### in frequency domain, multiply a certain factor
                     ###convolution
    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT))) ## y = dirac convolution V

    dydxFFT = 1.j*kGrid*yFFT #1.j=1j 1st derivative
    dydx = np.real(np.fft.ifft(np.fft.ifftshift(dydxFFT)))

    return xGrid, y, dydx ##y energy dydx force

def computeLJ_FPotPer(Nx, mu, Ls, cutIdx = 50, xCenter = 0, nPointSmear = 10):

    xGrid = np.linspace(0, Ls, Nx+1)[:-1]
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = nPointSmear*Ls/Nx

    x = gaussian(xGrid, xCenter, tau) + \
        gaussian(xGrid - Ls, xCenter, tau) +\
        gaussian(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x))

    yFFT = xFFT*mult

    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

    dydxFFT = 1.j*kGrid*yFFT
    dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

    potPer = potentialLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
             potentialLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
             potentialLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])

    derPotPer = forcesLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
                forcesLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
                forcesLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])

    y = y + potPer
    dydx = np.real(dydx) + derPotPer

    return xGrid, y, dydx


def potentialLJ(x,y, epsilon, sigma):
    return 4*epsilon*(pow(sigma/np.abs(x-y), 12) - pow(sigma/np.abs(x-y), 6) )

def forcesLJ(x,y, epsilon, sigma):
    return -4*epsilon/sigma*(-12*np.sign(x-y)*pow(sigma/np.abs(x-y), 13) + \
                        6*np.sign(x-y)*pow(sigma/np.abs(x-y), 7) )

def genDataLJ_FPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))

    if Lcell == 0.0 :
        sizeCell = 1/Ncells
    else :
        sizeCell = Lcell

    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell
    cutIdx = round(Nx*minDelta/Ls)

    xGrid, pot, dpotdx = computeLJ_FPotPer(Nx, sigma, Ls, cutIdx = cutIdx)

    # this is the old version
    # idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)

    idxCell = np.linspace(cutIdx//2,NpointsPerCell-1-cutIdx//2, NpointsPerCell- 2*(cutIdx//2)).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)

    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
        points = xGrid[idxPointCell]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])


        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

        pointsArray[i, :] = points.T

        R = pot[idxPointCell - idxPointCell.T]

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = dpotdx[idxPointCell - idxPointCell.T]
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = -np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T

    return pointsArray, potentialArray, forcesArray



# this doesn't really work due to the Gibbs phenomenon.
def computeDerPotPerNUFFT(Nx, mu, Ls, xCenter = 0):

    xGrid = np.linspace(0, Ls, Nx+1)[:-1]
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = 12*(Ls/(2*np.pi*Nx))**2

    x = gaussianNUFFT(xGrid, xCenter, tau) + \
        gaussianNUFFT(xGrid - Ls, xCenter, tau) +\
        gaussianNUFFT(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x))

    filterDeconv = gaussianDeconv(kGrid, tau)

    xFFTDeconv = xFFT*filterDeconv
    yFFT = xFFTDeconv*mult/(2*np.pi*Nx/Ls)

    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

    dydxFFT = 1.j*kGrid*yFFT
    dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

    return xGrid, y, np.real(dydx)


def genDataYukawaPerIndex(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0):

    pointsArray = np.zeros((Nsamples, Np*Ncells))
    idxArray = np.zeros((Nsamples, Ncells, Np)).astype(int)
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))


    if Lcell == 0.0 :
        sizeCell = 1/Ncells #calculate a density
    else :
        sizeCell = Lcell

    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell

    xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
#    idxArray = np.zeros((Nsamples, Ncells,Np))
    for i in range(Nsamples):

        idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
        idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0) ##position
        points = xGrid[idxPointCell1]
        # this is to keep the periodicity
        pointsExt = np.concatenate([points - Ls, points, points + Ls])
        # we want to check that the points are not too close
        while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
            idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
            idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
            points = xGrid[idxPointCell1]
            pointsExt = np.concatenate([points - Ls, points, points + Ls])

#        idxArray[i,:,:] = idxPointCell
        pointsArray[i, :] = points.T
        idxArray[i,:,:] = idxPointCell

        R = pot[idxPointCell1 - idxPointCell1.T]##matrix interaction

        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        F = dpotdx[idxPointCell1 - idxPointCell1.T]
        F = np.triu(F,1) + np.tril(F,-1)

        Forces = -np.sum(F, axis = 1)

        forcesArray[i,:] = Forces.T
        
#    idxArray = idxArray.astype(int)
    return idxArray, pointsArray, potentialArray, forcesArray


#def computeEFfromIndex(idxArray, Ncells, mu, sizeCell, xCenter = 0, nPointSmear = 10):
def computeEFfromIndex(idxArray, Ncells, mu, sizeCell, xCenter = 0, nPointSmear = 10):
    ######idxArray (Nsamples , Ncells , Np)
    ######for later index, we need to reduce them into the domain (shift)
    Nx = Ncells*1000 + 1
    Ls = Ncells*sizeCell
    xGrid = np.linspace(0, Ls, Nx+1)[:-1] ##delete the last one
    kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls ##frequency domain

    filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
    mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))
    ##yukuwa potential V = F(mult)

    # here we smear the dirac delta
    # we use the width of the smearing for
    tau = nPointSmear*Ls/Nx #standard deviation for guass, use guass to approach dirac

    x = gaussian(xGrid, xCenter, tau) + \
        gaussian(xGrid - Ls, xCenter, tau) +\
        gaussian(xGrid + Ls, xCenter, tau)

    xFFT = np.fft.fftshift(np.fft.fft(x)) #FFT of delta function, shift is just for convenience

    yFFT = xFFT*mult ### in frequency domain, multiply a certain factor
                     ###convolution
    y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT))) ## y = dirac convolution V

    dydxFFT = 1.j*kGrid*yFFT #1.j=1j 1st derivative
    dydx = np.real(np.fft.ifft(np.fft.ifftshift(dydxFFT)))


    Nsamples = idxArray.shape[0]
    Np = idxArray.shape[2]
#    pointsArray = np.zeros((Nsamples, Np*Ncells))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells))  
    pointsArray = np.zeros((Nsamples,Np*Ncells))
    
    for j in range(Nsamples):
        idxPointCell = idxArray[j,:,:]
#        idxPointCell = idxArray[j,:,:]
        idxPointCell1 = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
        points = xGrid[idxPointCell1]
        pointsArray[j, :] = points.T
        
        R = y[idxPointCell1 - idxPointCell1.T]##matrix interaction
        RR = np.triu(R, 1) ###consider interaction effect
        potTotal = np.sum(RR)
        potentialArray[j,:] = potTotal
        F = dydx[idxPointCell1 - idxPointCell1.T]
        F = np.triu(F,1) + np.tril(F,-1)
        Forces = -np.sum(F, axis = 1)
        forcesArray[j,:] = Forces.T
        
    return pointsArray, potentialArray, forcesArray