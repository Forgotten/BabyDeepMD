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


# TODO: modify this for 3D
def genDataPer3D(Ncells, Np, mu, Nsamples, minDelta = 0.0, Lcell = 0.0): 

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

		R = potentialPer(points, pointsT, mu, L)

		RR = np.triu(R, 1)
		potTotal = np.sum(RR)

		potentialArray[i,:] = potTotal

		F = forcesPer(points,pointsT, mu, L)

		Forces = np.sum(F, axis = 1) 

		forcesArray[i,:,:] = np.reshape(Forces,(Np*Ncells**2, 2))

	return pointsArray, potentialArray, forcesArray


def gaussian3D(x, y, z, center, tau):
	return (1/sqrt(2*np.pi*tau)**3)*\
		   np.exp( -0.5*(  np.square(x - center[0])
		   	             + np.square(y - center[1])
		   	             + np.square(z - center[2]))/tau**2 )


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

