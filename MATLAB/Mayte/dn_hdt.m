function f = dn_hdt(V,n_ih,P) 
%      n_ih_inf = @(V) 1./(1 + exp((V-P.V_half)/P.k));
    n_ih_inf = @(V) ones(size(V)).*(V<=P.V_m)+...
        (1/2 - (V-P.V_half)/(4*P.k)).*(V>P.V_m & V<P.V_p)+...
        zeros(size(V)).*(V>=P.V_p);
    f = (n_ih_inf(V) - n_ih)/P.tau_h;
end