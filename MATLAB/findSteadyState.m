% Find steady state of single neuron

%% Parameter values from Fig. 4 in paper
clear; close all;

% thresholds
V_half  = -10.0;
V_k     = 10.0;
V_left  = V_half-2*V_k;
V_right = V_half+2*V_k;
V_r     = 0.0;
V_th    = 14.0;
V_h     = 40.0;

alpha       = 0.05;
beta_left   = 1.0;
beta_right  = 0.0;
tau  = 1.0;
tau_h= 400.0;
tau_r = 20.0;
gh   = 1.0;
gs   = 15.0;
gl   = 0.25;
I    = -30.0;
sigma= 25.0;
W    = 10.0;
beta_centre  = 0.5+V_half/(4*V_k);
gamma_centre = -1/(4*V_k);

% package parameters
dummy = who;
p     = struct;
for i = 1:length(dummy)
  p = setfield(p,dummy{i},eval(dummy{i}));
end

%% Now find steady state
options = optimset('Display','iter');
fun = @(u) IFfunSteve( 0, u, p);
u0 = fsolve( fun, [8.0;0.05;0;0],options);
fprintf('Steady state found at: V=%f, n=%f.\n',u0(1),u0(2));
