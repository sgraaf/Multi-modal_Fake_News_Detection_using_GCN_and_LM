from scipy.io import loadmat
x = loadmat('BuzzFeedUserFeature.mat')

print(len(x))
print(type(x))
print(x.keys())

print(type(x['__header__']))	
print(type(x['__version__']))	
print(x['__version__'])
print(type(x['__globals__']))	
print(len(x['__globals__']))	
print(x['__globals__'])	

print(type(x['X']))	
print(x['X'].get_shape())
print(x['X'].getrow(0))
print(x['X'].getrow(1))
print(x['X'].getrow(10000))
print(x['X'].getrow(0).getnnz())
print(x['X'].getrow(1).getnnz())
print(x['X'].getrow(0).getcol(225))
print(x['X'].getrow(1).getcol(226))

print(x['X'].get_shape())
# number of users by 