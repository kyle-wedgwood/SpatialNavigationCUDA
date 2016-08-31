function f = dudt(u,y,P) 
    f = (y-u)*P.alpha;
end