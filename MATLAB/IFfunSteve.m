function dydt = IFfunSteve( ~, state, p)

  % unpack variables
  v = state(1);
  n = state(2);
  u = state(3);
  y = state(4);

  % unpack parameters
  alpha        = p.alpha;
  tau          = p.tau;
  tau_h        = p.tau_h;
  gs           = p.gs;
  gh           = p.gh;
  gl           = p.gl;
  beta_left    = p.beta_left;
  beta_right   = p.beta_right;
  I            = p.I;
  gamma_centre = p.gamma_centre;
  beta_centre  = p.beta_centre;
  V_left       = p.V_left;
  V_right      = p.V_right;
  V_h          = p.V_h;

  % calculate ninf
  ninf = beta_left*(v<V_left)...
         +(gamma_centre*v+beta_centre).*(V_left<v).*(v<V_right)...
         +beta_right*(V_right<v);

  dv = 1/tau*(-gl*v+gh*V_h*n+gs*u+I);
  dn = (ninf-n)/tau_h;
  du = alpha*(y-u);
  dy = -alpha*y;

  % pack
  dydt = [dv;dn;du;dy];

end
