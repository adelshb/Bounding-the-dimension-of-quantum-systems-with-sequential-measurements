function objective = I3322NV2014(d)

 % find maximal value of I3322 with Navascues Vertesi 2014 method of random certificates;
    saved_gammas=load('paula');
    basis_gamma = saved_gammas.basis_gamma;
    l = length(basis_gamma);
    yalmip('clear');
    alpha = sdpvar(l,1,'full','real'); % coefficients
    Gamma = 0;
    for i=1:l
        fprintf('summing basis %f\n',100*i/l)
        Gamma = Gamma + alpha(i)*basis_gamma{i};
    end    
    %% objective

coeff = [0  1  1  0;
	 1 -1 -1  1;
	 1 -1 -1 -1;
	 0  1 -1  0;]; 	% i3322

    inputA=3;
    inputB=3;

    marginal = 0;
    for x=0:inputA-1
        marginal = marginal + coeff(x+2,1)*Gamma(1,2+x);
    end
    for y=0:inputB-1
        marginal = marginal + coeff(1,y+2)*Gamma(1,2+inputA+y);
    end
    correlation = 0;
    for x=0:inputA-1
        for y=0:inputB-1
            correlation = correlation + coeff(x+2,y+2)*Gamma(2+x,2+inputA+y);
        end
    end

    F = [Gamma >= 0, Gamma(1,1) == 1]; % constraints

    %% solve it 

    objective = real(correlation + marginal);
    ops = sdpsettings(sdpsettings,'verbose',1,'solver','sedumi');
    sol = solvesdp(F,-objective,ops);
    objective = double(objective);
    correlation = double(correlation);
    marginal = double(marginal);

end
