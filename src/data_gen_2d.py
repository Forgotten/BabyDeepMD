import numpy as np

def potential(x,y, mu):
	return -np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1)))

def forces(x,y, mu):
	return -mu*(y - x)/(np.finfo(float).eps+np.sqrt(np.sum(np.square(y - x), \
													       axis = -1, keepdims = True)))\
		   *np.exp(-mu*np.sqrt(np.sum(np.square(y - x), axis = -1, keepdims = True)))

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
		while np.min( distPoints[distPoints>0] ) < minDelta:
			points = midPoints + sizeCell*(np.random.rand(Ncells, Ncells, Np,2)-0.5)
			distPoints = np.sqrt(np.sum(np.square(np.reshape(points, (-1,1,2)) 
											  -np.reshape(points, (1,-1,2))), axis=-1))

		pointsArray[i, :, :] = np.reshape(points,( Np*Ncells**2, 2))
		points = np.reshape(points, (Np*Ncells**2,1,2))
		pointsT = np.reshape(points, (1,Np*Ncells**2,2))

		R = potentialPer(points,pointsT, mu, L)

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
	mult = 4*np.pi*filterM/(np.square(kx_grid) + np.square(ky_grid) + np.square(mu))

	# here we smear the dirac delta
	# we use the width of the smearing for 
	tau = nPointSmear*Ls/Nx

	x = gaussian2D(x_grid-Ls, y_grid-Ls, xCenter, tau) + \
			gaussian2D(x_grid-Ls, y_grid   , xCenter, tau) + \
			gaussian2D(x_grid-Ls, y_grid+Ls, xCenter, tau) + \
			gaussian2D(x_grid		, y_grid-Ls, xCenter, tau) + \
			gaussian2D(x_grid   , y_grid   , xCenter, tau) + \
			gaussian2D(x_grid   , y_grid+Ls, xCenter, tau) + \
			gaussian2D(x_grid+Ls, y_grid-Ls, xCenter, tau) + \
			gaussian2D(x_grid+Ls, y_grid   , xCenter, tau) + \
			gaussian2D(x_grid+Ls, y_grid+Ls, xCenter, tau) 

	xFFT = np.fft.fftshift(np.fft.fft2(x))
	
	fFFT = xFFT*mult
	
	f = np.real(np.fft.ifft2(np.fft.ifftshift(fFFT)))

	dfdxFFT = 1.j*kx_grid*fFFT
	dfdyFFT = 1.j*ky_grid*fFFT

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

	idx_point_x = idx_start_x.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_x.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

	idx_point_y = idx_start_y.reshape((Ncells, Ncells, 1)) \
		              + np.random.choice(idx_cell_y.reshape((-1,)), 
		              	                 [Ncells, Ncells, Np])

	for i in range(Nsamples):

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

		Fy = dpotdx[idx_point_x.reshape((-1,1)) 
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



# def potentialGaussian(x, y, sigma):
# 	return -np.exp(-sigma*np.square(y - x))


# def forcesGaussian(x, y, sigma):
# 	return -sigma*2*(y - x)*np.exp(-sigma*np.square(y - x))


# def gen_dataGaussian(Ncells, Np, sigma, Nsamples, minDelta = 0.0): 

# 	pointsArray = np.zeros((Nsamples, Np*Ncells))
# 	potentialArray = np.zeros((Nsamples,1))
# 	forcesArray = np.zeros((Nsamples, Np*Ncells))

# 	for i in range(Nsamples):
# 		sizeCell = 1/Ncells
# 		midPoints = np.linspace(sizeCell/2.0,1-sizeCell/2.0, Ncells)

# 		points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
# 		points = np.sort(points.reshape((-1,1)), axis = 0)

# 		# we want to check that the points are not too close 
# 		while np.min(points[1:] - points[0:-1]) < minDelta:
# 			points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
# 			points = np.sort(points.reshape((-1,1)), axis = 0)

# 		pointsArray[i, :] = points.T

# 		R = potentialGaussian(points,points.T,sigma)

# 		RR = np.triu(R, 1)
# 		potTotal = np.sum(RR)

# 		potentialArray[i,:] = potTotal

# 		F = forcesGaussian(points,points.T,sigma)

# 		Forces = np.sum(F, axis = 1) 

# 		forcesArray[i,:] = Forces.T

# 	return pointsArray, potentialArray, forcesArray



# def potentialYukawa(x, y, mu):
# 	return -np.exp(-mu*np.abs(y - x))/np.abs(y - x)


# def forcesYukawa(x, y, mu):
# 	return 	mu*np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.abs(y - x) + np.sign(y - x)*np.exp(-mu*np.abs(y - x))/np.square(np.abs(y - x))


# def genDataYukawa(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

# 	pointsArray = np.zeros((Nsamples, Np*Ncells))
# 	potentialArray = np.zeros((Nsamples,1))
# 	forcesArray = np.zeros((Nsamples, Np*Ncells))

# 	for i in range(Nsamples):
# 		if Lcell == 0.0 :
# 			sizeCell = 1/Ncells
# 		else :
# 			sizeCell = Lcell

# 		midPoints = np.linspace(sizeCell/2.0,Ncells*sizeCell-sizeCell/2.0, Ncells)

# 		points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
# 		points = np.sort(points.reshape((-1,1)), axis = 0)

# 		# we want to check that the points are not too close 
# 		while np.min(points[1:] - points[0:-1]) < minDelta:
# 			points = midPoints + sizeCell*(np.random.rand(Np, Ncells) -0.5)
# 			points = np.sort(points.reshape((-1,1)), axis = 0)

# 		pointsArray[i, :] = points.T

# 		R = potentialYukawa(points,points.T,sigma)

# 		RR = np.triu(R, 1)
# 		potTotal = np.sum(RR)

# 		potentialArray[i,:] = potTotal

# 		F = forcesYukawa(points,points.T,sigma)
# 		F = np.triu(F,1) + np.tril(F,-1)

# 		Forces = np.sum(F, axis = 1) 

# 		forcesArray[i,:] = Forces.T

# 	return pointsArray, potentialArray, forcesArray


# def genDataYukawaPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

# 	pointsArray = np.zeros((Nsamples, Np*Ncells))
# 	potentialArray = np.zeros((Nsamples,1))
# 	forcesArray = np.zeros((Nsamples, Np*Ncells))

# 	if Lcell == 0.0 :
# 		sizeCell = 1/Ncells
# 	else :
# 		sizeCell = Lcell

# 	NpointsPerCell = 1000
# 	Nx = Ncells*NpointsPerCell + 1
# 	Ls = Ncells*sizeCell

# 	xGrid, pot, dpotdx = computeDerPotPer(Nx, sigma, Ls)

# 	idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)
# 	idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
# 	for i in range(Nsamples):

# 		idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
# 		idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
# 		points = xGrid[idxPointCell]
# 		# this is to keep the periodicity
# 		pointsExt = np.concatenate([points - Ls, points, points + Ls])
		

# 		# we want to check that the points are not too close 
# 		while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
# 			idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
# 			idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
# 			points = xGrid[idxPointCell]
# 			pointsExt = np.concatenate([points - Ls, points, points + Ls])

# 		pointsArray[i, :] = points.T

# 		R = pot[idxPointCell - idxPointCell.T]

# 		RR = np.triu(R, 1)
# 		potTotal = np.sum(RR)

# 		potentialArray[i,:] = potTotal

# 		F = dpotdx[idxPointCell - idxPointCell.T]
# 		F = np.triu(F,1) + np.tril(F,-1)

# 		Forces = -np.sum(F, axis = 1) 

# 		forcesArray[i,:] = Forces.T

# 	return pointsArray, potentialArray, forcesArray


# def gaussian(x, xCenter, tau):
# 	return (1/np.sqrt(2*np.pi*tau**2))*\
# 		   np.exp( -0.5*np.square(x - xCenter)/tau**2 )

# def gaussianNUFFT(x, xCenter, tau):
# 	return np.exp(-np.square(x - xCenter)/(4*tau) )

# def gaussianDeconv(k, tau):
# 	return np.sqrt(np.pi/tau)*np.exp( np.square(k)*tau)

# def computeDerPotPer(Nx, mu, Ls, xCenter = 0, nPointSmear = 10):   
	
# 	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
# 	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

# 	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
# 	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

# 	# here we smear the dirac delta
# 	# we use the width of the smearing for 
# 	tau = nPointSmear*Ls/Nx

# 	x = gaussian(xGrid, xCenter, tau) + \
# 		gaussian(xGrid - Ls, xCenter, tau) +\
# 		gaussian(xGrid + Ls, xCenter, tau) 

# 	xFFT = np.fft.fftshift(np.fft.fft(x))
	
# 	yFFT = xFFT*mult
	
# 	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

# 	dydxFFT = 1.j*kGrid*yFFT
# 	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

# 	return xGrid, y, np.real(dydx)

# def computeLJ_FPotPer(Nx, mu, Ls, cutIdx = 50, xCenter = 0, nPointSmear = 10):
	
# 	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
# 	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

# 	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
# 	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

# 	# here we smear the dirac delta
# 	# we use the width of the smearing for 
# 	tau = nPointSmear*Ls/Nx

# 	x = gaussian(xGrid, xCenter, tau) + \
# 		gaussian(xGrid - Ls, xCenter, tau) +\
# 		gaussian(xGrid + Ls, xCenter, tau) 

# 	xFFT = np.fft.fftshift(np.fft.fft(x))
	
# 	yFFT = xFFT*mult
	
# 	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

# 	dydxFFT = 1.j*kGrid*yFFT
# 	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

# 	potPer = potentialLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
# 			 potentialLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
# 			 potentialLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])   

# 	derPotPer = forcesLJ(xGrid,    xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
# 				forcesLJ(xGrid-Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx]) + \
# 				forcesLJ(xGrid+Ls, xCenter, 2*y[cutIdx], xGrid[cutIdx])   

# 	y = y + potPer
# 	dydx = np.real(dydx) + derPotPer

# 	return xGrid, y, dydx


# def potentialLJ(x,y, epsilon, sigma):
# 	return 4*epsilon*(pow(sigma/np.abs(x-y), 12) - pow(sigma/np.abs(x-y), 6) )

# def forcesLJ(x,y, epsilon, sigma):
# 	return 4*epsilon*(-12*sigma*np.sign(x-y)*pow(sigma/np.abs(x-y), 11) + \
# 						6*sigma*np.sign(x-y)*pow(sigma/np.abs(x-y), 5) )

# def genDataLJ_FPer(Ncells, Np, sigma, Nsamples, minDelta = 0.0, Lcell = 0.0): 

# 	pointsArray = np.zeros((Nsamples, Np*Ncells))
# 	potentialArray = np.zeros((Nsamples,1))
# 	forcesArray = np.zeros((Nsamples, Np*Ncells))

# 	if Lcell == 0.0 :
# 		sizeCell = 1/Ncells
# 	else :
# 		sizeCell = Lcell

# 	NpointsPerCell = 1000
# 	Nx = Ncells*NpointsPerCell + 1
# 	Ls = Ncells*sizeCell
# 	cutIdx = round(Nx*minDelta/Ls)

# 	xGrid, pot, dpotdx = computeLJ_FPotPer(Nx, sigma, Ls, cutIdx = cutIdx)

# 	# this is the old version 
# 	# idxCell = np.linspace(0,NpointsPerCell-1, NpointsPerCell).astype(int)

# 	idxCell = np.linspace(cutIdx//2,NpointsPerCell-1-cutIdx//2, NpointsPerCell- 2*(cutIdx//2)).astype(int)
# 	idxStart = np.array([ii*NpointsPerCell for ii in range(Ncells)]).reshape(-1,1)
	
# 	for i in range(Nsamples):

# 		idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
# 		idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
# 		points = xGrid[idxPointCell]
# 		# this is to keep the periodicity
# 		pointsExt = np.concatenate([points - Ls, points, points + Ls])
		

# 		# we want to check that the points are not too close 
# 		while np.min(pointsExt[1:] - pointsExt[0:-1]) < minDelta:
# 			idxPointCell = idxStart + np.random.choice(idxCell, [Ncells, Np  ])
# 			idxPointCell = np.sort(idxPointCell.reshape((-1,1)), axis = 0)
# 			points = xGrid[idxPointCell]
# 			pointsExt = np.concatenate([points - Ls, points, points + Ls])

# 		pointsArray[i, :] = points.T

# 		R = pot[idxPointCell - idxPointCell.T]

# 		RR = np.triu(R, 1)
# 		potTotal = np.sum(RR)

# 		potentialArray[i,:] = potTotal

# 		F = dpotdx[idxPointCell - idxPointCell.T]
# 		F = np.triu(F,1) + np.tril(F,-1)

# 		Forces = -np.sum(F, axis = 1) 

# 		forcesArray[i,:] = Forces.T

# 	return pointsArray, potentialArray, forcesArray



# # this doesn't really work due to the Gibbs phenomenon. 
# def computeDerPotPerNUFFT(Nx, mu, Ls, xCenter = 0):
	
# 	xGrid = np.linspace(0, Ls, Nx+1)[:-1] 
# 	kGrid = 2*np.pi*np.linspace(-(Nx//2), Nx//2, Nx)/Ls      

# 	filterM = 1#0.5 - 0.5*np.tanh(np.abs(3*kGrid/np.sqrt(Nx)) - np.sqrt(Nx)) 
# 	mult = 4*np.pi*filterM/(np.square(kGrid) + np.square(mu))

# 	# here we smear the dirac delta
# 	# we use the width of the smearing for 
# 	tau = 12*(Ls/(2*np.pi*Nx))**2 

# 	x = gaussianNUFFT(xGrid, xCenter, tau) + \
# 		gaussianNUFFT(xGrid - Ls, xCenter, tau) +\
# 		gaussianNUFFT(xGrid + Ls, xCenter, tau) 

# 	xFFT = np.fft.fftshift(np.fft.fft(x))

# 	filterDeconv = gaussianDeconv(kGrid, tau)

# 	xFFTDeconv = xFFT*filterDeconv
# 	yFFT = xFFTDeconv*mult/(2*np.pi*Nx/Ls)
	
# 	y = np.real(np.fft.ifft(np.fft.ifftshift(yFFT)))

# 	dydxFFT = 1.j*kGrid*yFFT
# 	dydx = np.fft.ifft(np.fft.ifftshift(dydxFFT))

# 	return xGrid, y, np.real(dydx)
