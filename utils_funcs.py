from functools import lru_cache
model_config=None
def getk(i):
    k=i
    nblock_per_stage=(model_config['depth']-2)//6
    i=(k-1)//(nblock_per_stage*2)
    return model_config["stage%d"%(i+1)]['k%ds'%((k+1)%2+1)][((k-1)%(nblock_per_stage*2))//2]

def gets(i):
    k=i
    if k%2==1:
        return 1
    nblock_per_stage=(model_config['depth']-2)//6
    i=(k-1)//(nblock_per_stage*2)
    if (((k-1)%(nblock_per_stage*2))//2 + 1) in set(model_config["stage%d"%(i+1)]['maxpool']):
        return 2
    return 1
@lru_cache(maxsize=None)
def get_maxrf(i):
    if i==0:
        return 2,5 # starting RF
    s,rf=get_maxrf(i-1)
    s=s*gets(i)
    rf= rf+ (getk(i)-1)*s
    return s,rf
