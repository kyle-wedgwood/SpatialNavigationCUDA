function flag = checkEvents(Y,zone,N,V_left,V_right,V_th)

  v1 = Y(1:N);
  r1 = Y(4*N+1:5*N);
  
  flag = (v1>V_left) & (zone==1);
  flag = flag+2*( (v1<V_left) & (zone==2) );
  flag = flag+3*( (v1>V_right) & (zone==2) );
  flag = flag+4*( (v1<V_right) & (zone==3) );
  flag = flag+5*( (v1>V_th) & (zone==3) );
  flag = flag+6*( (r1<0.0) & (zone==4) );
  
end