import numpy as np

def potential(x,y, mu):
	return -np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1)))
#### exp(-mu*|x-y|) |x-y|=sqrt(sum(x-y)**2)
def forces(x,y, mu):
	return -mu*(y - x)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(y - x), \
													       axis = -1, keepdims = True)))\
		   *np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1, keepdims = True)))
###compute the gradient 
def potentialPer(x,y, mu, L):
	shift_x = np.reshape(np.array([L, 0.]), (1,1,2))
	shift_y = np.reshape(np.array([0., L]), (1,1,2))

	return potential(x,y, mu) + potential(x+shift_x,y, mu) + potential(x-shift_x,y, mu)\
		   +potential(x+shift_y,y, mu) + potential(x+shift_x+shift_y,y, mu) + potential(x-shift_x+shift_y,y, mu) \
		   +potential(x-shift_y,y, mu) + potential(x+shift_x-shift_y,y, mu) + potential(x-shift_x-shift_y,y, mu)


def forcesPer(x,y, mu, L):
	shift_x = np.reshape(np.array([L, 0.]), (1,1,2))
	shift_y = np.reshape(np.array([0., L]), (1,1,2))

	return   forces(x,y, mu) + forces(x+shift_x,y, mu) + forces(x-shift_x,y, mu)\
		   + forces(x+shift_y,y, mu) + forces(x+shift_x+shift_y,y, mu) + forces(x-shift_x+shift_y,y, mu)\
		   + forces(x-shift_y,y, mu) + forces(x+shift_x-shift_y,y, mu) + forces(x-shift_x-shift_y,y, mu)


def genDataPer2D(Ncells, Np, mu, Nsamples, minDelta = 0.0, Lcell = 0.0): 

	pointsArray = np.zeros((Nsamples, Np*Ncells**2, 2))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells**2, 2))


	if Lcell == 0.0 :
		sizeCell = 1/Ncells
	else :
		sizeCell = Lcell

	L = sizeCell*Ncells

	midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

	xx, yy = np.meshgrid(midPoints, midPoints)

	midPoints = np.concatenate([np.reshape(xx, (Ncells,Ncells,1,1)), 
								np.reshape(yy, (Ncells,Ncells,1,1))], axis = -1) 

	for i in range(Nsamples):

		points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2) -0.5)

		relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))

		relPointsPer = relPoints - L*np.round(relPoints/L)

		distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))

		# we want to check that the points are not too close 
        ### here is a small adjustment 
		while np.min( distPoints[distPoints>0] ) < minDelta:
		    points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2)-0.5)
		    relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))
    
		    relPointsPer = relPoints - L*np.round(relPoints/L)
    
		    distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))            
            
#			distPoints = np.sqrt(np.sum(np.square(np.reshape(points, (-1,1,2)) 
#											  -np.reshape(points, (1,-1,2))), axis=-1))

		pointsArray[i, :, :] = np.reshape(points,( Np*Ncells**2, 2))
		points = np.reshape(points, (Np*Ncells**2,1,2))
		pointsT = np.reshape(points, (1,Np*Ncells**2,2))

		R = potentialPer(points, pointsT, mu, L)

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = forcesPer(points,pointsT, mu, L)

		Forces = np.sum(F, axis = 1)

		forcesArray[i,:,:] = np.reshape(Forces,(Np*Ncells**2, 2))

	return pointsArray, potentialArray, forcesArray


def gaussian2D(x,y, center, tau):
	return (1/(2*np.pi*tau**2))*\
		   np.exp( -0.5*(  np.square(x - center[0])
		   	             + np.square(y - center[1]))/tau**2 )


def computeDerPot2DPer(Nx, mu, Ls, xCenter = [0.0, 0.0], nPointSmear = 10):   
	
	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

	# creating the 2D space and frequency grids
	y_grid, x_grid = np.meshgrid(xGrid, xGrid)
	ky_grid, kx_grid = np.meshgrid(kGrid, kGrid)


	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx))
	# multiplier 
	mult = ((Nx/Ls)**2)*4*np.pi*filterM/(np.square(kx_grid) + np.square(ky_grid) + np.square(mu))

	# here we smear the dirac delta
	# we use the width of the smearing for 
#	tau = nPointSmear*Ls/Nx
#
#	x = gaussian2D(x_grid-Ls, y_grid-Ls, xCenter, tau) + \
#			gaussian2D(x_grid-Ls, y_grid   , xCenter, tau) + \
#			gaussian2D(x_grid-Ls, y_grid+Ls, xCenter, tau) + \
#			gaussian2D(x_grid		, y_grid-Ls, xCenter, tau) + \
#			gaussian2D(x_grid   , y_grid   , xCenter, tau) + \
#			gaussian2D(x_grid   , y_grid+Ls, xCenter, tau) + \
#			gaussian2D(x_grid+Ls, y_grid-Ls, xCenter, tau) + \
#			gaussian2D(x_grid+Ls, y_grid   , xCenter, tau) + \
#			gaussian2D(x_grid+Ls, y_grid+Ls, xCenter, tau) 
#
#	xFFT = np.fft.fftshift(np.fft.fft2(x))
#	
#	fFFT = xFFT*mult
	
	f = np.real(np.fft.ifft2(np.fft.ifftshift(mult)))

	dfdxFFT = 1.j*kx_grid*mult
	dfdyFFT = 1.j*ky_grid*mult

	dfdx = np.fft.ifft2(np.fft.ifftshift(dfdxFFT))
	dfdy = np.fft.ifft2(np.fft.ifftshift(dfdyFFT))


	return x_grid, y_grid, f, np.real(dfdx), np.real(dfdy)



def genDataYukawa2DPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

    pointsArray = np.zeros((Nsamples, Np*Ncells**2, 2))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells**2, 2))

    if Lcell == 0.0 :
        sizeCell = 1/Ncells
    else :
        sizeCell = Lcell

    NpointsPerCell = 1000
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell

    x_grid, y_grid, pot, dpotdx, dpotdy = computeDerPot2DPer(Nx, sigma, Ls)

    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
    idx_cell_y, idx_cell_x = np.meshgrid(idxCell, idxCell) 
    idx_start_y, idx_start_x = np.meshgrid(idxStart, idxStart) 

#	idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
#		              + np.random.choice(idx_cell_x.reshape((-1,)), 
#		              	                 [Ncells, Ncells, Np])
#
#	idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
#		              + np.random.choice(idx_cell_y.reshape((-1,)), 
#		              	                 [Ncells, Ncells, Np])

    for i in range(Nsamples):
        idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1))+ np.random.choice(idx_cell_x.reshape((-1,)), 
    		              	                 [Ncells, Ncells, Np])
    
        idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1))+ np.random.choice(idx_cell_y.reshape((-1,)), 
    		              	                 [Ncells, Ncells, Np])
		# just starting point 
        dist = [10.0, 10.0]

        while np.min(dist) < minDelta:

            idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_x.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

            idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_y.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

            points_x = x_grid[idx_point_x, 0]
            points_y = y_grid[0, idx_point_y]

			# we compute the periodic distance

            diff_x = points_x.reshape((-1,1)) - points_x.reshape((-1,1)).T
            diff_y = points_y.reshape((-1,1)) - points_y.reshape((-1,1)).T

			# periodization
            diff_x -= Ls*np.round(diff_x/Ls)
            diff_y -= Ls*np.round(diff_y/Ls)

			# computing the distance
            dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
            dist += 10*np.eye(Ncells**2*Np)


        pointsArray[i, :, 0] = x_grid[idx_point_x.reshape((-1,)), 0]
        pointsArray[i, :, 1] = y_grid[0, idx_point_y.reshape((-1,))]

        R = pot[idx_point_x.reshape((-1,1)) 
						- idx_point_x.reshape((-1,1)).T, 
						idx_point_y.reshape((-1,1)) 
						- idx_point_y.reshape((-1,1)).T]

        RR = np.triu(R, 1)
        potTotal = np.sum(RR)

        potentialArray[i,:] = potTotal

        Fx = dpotdx[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]

        Fy = dpotdy[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]
						   				   
        Fx = np.triu(Fx,1) + np.tril(Fx,-1)
        Fy = np.triu(Fy,1) + np.tril(Fy,-1)

        Forcesx = -np.sum(Fx, axis = 1) 
        Forcesy = -np.sum(Fy, axis = 1) 

        forcesArray[i,:,0] = Forcesx.T
        forcesArray[i,:,1] = Forcesy.T

    return pointsArray, potentialArray, forcesArray


def genDataYukawa2DPerMixed(Ncells, Np, sigma1,sigma2, Nsamples, minDelta = 0.0, Lcell = 0.0, weight1=0.5,weight2=0.5): 

    pointsArray = np.zeros((Nsamples, Np*Ncells**2, 2))
    potentialArray = np.zeros((Nsamples,1))
    forcesArray = np.zeros((Nsamples, Np*Ncells**2, 2))

    if Lcell == 0.0:
        sizeCell = 1/Ncells
    else:
        sizeCell = Lcell
        
    NpointsPerCell = 1000
    #NpointsPerCell = int(1000*sizeCell)
    Nx = Ncells*NpointsPerCell + 1
    Ls = Ncells*sizeCell

    x_grid, y_grid, pot1, dpotdx1, dpotdy1 = computeDerPot2DPer(Nx, sigma1, Ls)
    x_grid, y_grid, pot2, dpotdx2, dpotdy2 = computeDerPot2DPer(Nx, sigma2, Ls)
    idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
    idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
    idx_cell_y, idx_cell_x = np.meshgrid(idxCell, idxCell) 
    idx_start_y, idx_start_x = np.meshgrid(idxStart, idxStart) 

#    idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
#		              + np.random.choice(idx_cell_x.reshape((-1,)), 
#		              	                 [Ncells, Ncells, Np])
#
#    idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
#		              + np.random.choice(idx_cell_y.reshape((-1,)), 
#		              	                 [Ncells, Ncells, Np])

    for i in range(Nsamples):
        idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
    		              + np.random.choice(idx_cell_x.reshape((-1,)), 
    		              	                 [Ncells, Ncells, Np])
    
        idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
    		              + np.random.choice(idx_cell_y.reshape((-1,)), 
    		              	                 [Ncells, Ncells, Np])
		# just starting point 
        dist = [10.0, 10.0]

        while np.min(dist) < minDelta:

            idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_x.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

            idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_y.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

            points_x = x_grid[idx_point_x, 0]
            points_y = y_grid[0, idx_point_y]

			# we compute the periodic distance

            diff_x = points_x.reshape((-1,1)) - points_x.reshape((-1,1)).T
            diff_y = points_y.reshape((-1,1)) - points_y.reshape((-1,1)).T

			# periodization
            diff_x -= Ls*np.round(diff_x/Ls)
            diff_y -= Ls*np.round(diff_y/Ls)

			# computing the distance
            dist = np.sqrt(np.square(diff_x) + np.square(diff_y))
            dist += 10*np.eye(Ncells**2*Np)


        pointsArray[i, :, 0] = x_grid[idx_point_x.reshape((-1,)), 0]
        pointsArray[i, :, 1] = y_grid[0, idx_point_y.reshape((-1,))]

        R1 = pot1[idx_point_x.reshape((-1,1)) 
						- idx_point_x.reshape((-1,1)).T, 
						idx_point_y.reshape((-1,1)) 
						- idx_point_y.reshape((-1,1)).T]

        RR1 = np.triu(R1, 1)
        potTotal1 = np.sum(RR1)

        R2 = pot2[idx_point_x.reshape((-1,1)) 
						- idx_point_x.reshape((-1,1)).T, 
						idx_point_y.reshape((-1,1)) 
						- idx_point_y.reshape((-1,1)).T]

        RR2 = np.triu(R2, 1)
        potTotal2 = np.sum(RR2)
        potentialArray[i,:] = weight1*potTotal1 + weight2*potTotal2

        Fx1 = dpotdx1[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]

        Fy1 = dpotdy1[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]
						   				   
        Fx1 = np.triu(Fx1,1) + np.tril(Fx1,-1)
        Fy1 = np.triu(Fy1,1) + np.tril(Fy1,-1)

        Forcesx1 = -np.sum(Fx1, axis = 1) 
        Forcesy1 = -np.sum(Fy1, axis = 1) 

        Fx2 = dpotdx2[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]

        Fy2 = dpotdy2[idx_point_x.reshape((-1,1)) 
						   - idx_point_x.reshape((-1,1)).T, 
						   idx_point_y.reshape((-1,1)) 
						   - idx_point_y.reshape((-1,1)).T]
						   				   
        Fx2 = np.triu(Fx2,1) + np.tril(Fx2,-1)
        Fy2 = np.triu(Fy2,1) + np.tril(Fy2,-1)

        Forcesx2 = -np.sum(Fx2, axis = 1) 
        Forcesy2 = -np.sum(Fy2, axis = 1) 
        
        forcesArray[i,:,0] = weight1*Forcesx1.T + weight2*Forcesx2.T 
        forcesArray[i,:,1] = weight1*Forcesy1.T + weight2*Forcesy2.T 

    return pointsArray, potentialArray, forcesArray

def genDataPer2DMixed(Ncells, Np, mu1, mu2, Nsamples, minDelta = 0.0, Lcell = 0.0, weight1=0.5, weight2=0.5): 

	pointsArray = np.zeros((Nsamples, Np*Ncells**2, 2))
	potentialArray = np.zeros((Nsamples,1))
	forcesArray = np.zeros((Nsamples, Np*Ncells**2, 2))


	if Lcell == 0.0 :
		sizeCell = 1/Ncells
	else :
		sizeCell = Lcell

	L = sizeCell*Ncells

	midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

	xx, yy = np.meshgrid(midPoints, midPoints)

	midPoints = np.concatenate([np.reshape(xx, (Ncells,Ncells,1,1)), 
								np.reshape(yy, (Ncells,Ncells,1,1))], axis = -1) 

	for i in range(Nsamples):

		points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2) -0.5)

		relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))

		relPointsPer = relPoints - L*np.round(relPoints/L)

		distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))

		# we want to check that the points are not too close 
        ### here is a small adjustment 
		while np.min( distPoints[distPoints>0] ) < minDelta:
		    points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2)-0.5)
		    relPoints = np.reshape(points, (-1,1,2)) -np.reshape(points, (1,-1,2))
    
		    relPointsPer = relPoints - L*np.round(relPoints/L)
    
		    distPoints = np.sqrt(np.sum(np.square(relPointsPer), axis=-1))            
            
#			distPoints = np.sqrt(np.sum(np.square(np.reshape(points, (-1,1,2)) 
#											  -np.reshape(points, (1,-1,2))), axis=-1))

		pointsArray[i, :, :] = np.reshape(points,( Np*Ncells**2, 2))
		points = np.reshape(points, (Np*Ncells**2,1,2))
		pointsT = np.reshape(points, (1,Np*Ncells**2,2))

		R1 = potentialPer(points, pointsT, mu1, L)
		RR1 = np.triu(R1, 1)
		potTotal1 = np.sum(RR1)

		R2 = potentialPer(points, pointsT, mu2, L)
		RR2 = np.triu(R2, 1)
		potTotal2 = np.sum(RR2)
        
		potentialArray[i,:] = potTotal1*weight1 + potTotal2*weight2

		F1 = forcesPer(points,pointsT, mu1, L)
		Forces1 = np.sum(F1, axis = 1)

		F2 = forcesPer(points,pointsT, mu2, L)
		Forces2 = np.sum(F2, axis = 1)
        
		forcesArray[i,:,:] = np.reshape(Forces1,(Np*Ncells**2, 2))*weight1 +\
                             np.reshape(Forces2,(Np*Ncells**2, 2))*weight2

	return pointsArray, potentialArray, forcesArray

