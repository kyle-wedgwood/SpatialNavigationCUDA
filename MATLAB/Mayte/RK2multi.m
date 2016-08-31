function [V,n_ih,u,y] = RK2multi(V,n_ih,u,y,I_ini,delta_t,P)
    k1_V = delta_t*dVdt(V,n_ih,u,I_ini,P);
    k1_n_ih = delta_t*dn_hdt(V,n_ih,P);
    k1_u = delta_t*dudt(u,y,P);
    k1_y = delta_t*dydt(y,P);
    
    k2_V = delta_t*dVdt(V+delta_t*k1_V/2,n_ih+delta_t*k1_n_ih/2,u+delta_t*k1_u/2,I_ini,P);
    k2_n_ih = delta_t*dn_hdt(V+delta_t*k1_V/2,n_ih+delta_t*k1_n_ih/2,P);
    k2_u = delta_t*dudt(u+delta_t*k1_u/2,y+delta_t*k1_y/2,P);
    k2_y = delta_t*dydt(y+delta_t*k1_y/2,P);
    
    V = V + k2_V;
    n_ih = n_ih + k2_n_ih;
    u = u + k2_u;
    y = y + k2_y;
    
end