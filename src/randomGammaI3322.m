function G = randomGammaI3322(d) % generate random certificate for i3322
    
    psi = random_state_pure_vector(d^2); %in the beginning there was psi
    pvec = {eye(d^2)}; % initialize first cell with identity
    good_spectrum  = [ones(ceil(d/2),1); -ones(floor(d/2),1)];
%%level 1
    for i=2:4 % Alice's observables
	u = random_unitary(d);
        pvec{i} = kron(u*diag(good_spectrum)*u',eye(d));
    end
    for i=5:7 % Bob's observables
	u = random_unitary(d);
        pvec{i} = kron(eye(d),u*diag(good_spectrum)*u');
    end
    ind = 8;
%%level 2
    for i=2:7
        for j=2:7
            pvec{ind} = pvec{i}*pvec{j};
            ind = ind+1;
        end
    end
%%level 3
%    for i=2:7
%	for j=8:43
%	    pvec{ind} = pvec{i}*pvec{j};
%	    ind = ind+1;
%       end
%    end
%%level 4
%    for i=2:7
%	for j=44:259
%	    pvec{ind} = pvec{i}*pvec{j};
%	    ind = ind+1;
%       end
%    end
    d_gamma = ind-1;
    G = zeros(d_gamma,d_gamma); % initialize gamma matrix
    for i=1:d_gamma
        for j=i:d_gamma
            G(i,j) = real(psi'*pvec{i}'*pvec{j}*psi); % fill gamma matrix
        end
    end
    G = G + G' - diag(diag(G));
end
