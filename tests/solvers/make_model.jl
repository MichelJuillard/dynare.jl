function make_model(n)
    lli = [0     0     1     2     0     3;
           4     5     6     7     8     9;
           10    11     0     0     0    12]
    
    lli2 = repmat(lli,1,n)'
    lli2 = lli2[:]
    k = find(lli2')
    lli2[k] = 1:length(k)
    lli2 = reshape(lli2,n*6,3)'


    file = matopen("jacobian.mat")
    jacobian = read(file,"jacobia")
    jacobian2 = [kron(eye(n),jacobian[:,1:3]) kron(eye(n),jacobian[:,4:9]) kron(eye(n),jacobian[:,10:12])]
    return lli2, jacobian2
end
