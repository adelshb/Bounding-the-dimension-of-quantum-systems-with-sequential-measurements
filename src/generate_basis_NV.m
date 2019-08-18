function generate_basis_NV(d)

function final_kraus = gram_schmidt_good(kraus)
vec = cell(length(kraus),1);
size_base = 0;
for i=1:length(kraus)
        fprintf('completion0 %f\n',100*i/length(kraus))
	if norm(kraus{i}) <= 1e-6
		kraus{i} = 0;
	else
		kraus{i} = kraus{i}/sqrt(trace(kraus{i}'*kraus{i}));
		size_base = size_base + 1;
	   	vec{size_base} = kraus{i};
		for j=i+1:length(kraus)
			kraus{j} = kraus{j} - trace(kraus{i}'*kraus{j})*kraus{i};
		end
	end
end
final_kraus = cell(size_base,1);
final_kraus = vec(1:size_base);
end

%%start main loop

    d_gamma = length(randomGammaI3322(d));
    l = d_gamma*(d_gamma-1)/2;
%    l = 1500; % empirical
    basis_gamma = cell(l,1); % cells for random certificates
    rankmat = []; % matrix of Gammas in row-major order [like C], transposed
    for i=1:l
        fprintf('completion %f\n',100*i/l)
        basis_gamma{i} = randomGammaI3322(d); % generate random certificates
        rankmat = [rankmat; reshape(basis_gamma{i}',d_gamma^2,1)'];
	if rem(i,50) == 0
		dimension_basis = rank(rankmat);
		i,dimension_basis
		if dimension_basis < i
			break;
		end
	end
    end
    basis_gamma = basis_gamma(1:dimension_basis);
    save('paula','basis_gamma');
end
