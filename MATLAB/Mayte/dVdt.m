function  f = dVdt(V,n_ih,u,I_ini,P) 
    I_h = @(n_ih,P) P.g_h.*n_ih;
    I_syn = @(u,P) P.g_syn*u;
    f = (-P.g_l*V  + I_h(n_ih,P) + I_syn(u,P) + I_ini)/P.C;
end