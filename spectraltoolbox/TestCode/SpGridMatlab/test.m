%% Sparse Grid test

DIM = 10;
k = 5;

tic
[n,w] = nwspgr('GQN',DIM,k,1);
toc

save('nodesMat.txt','n','-ASCII')
save('weightsMat.txt','w','-ASCII')