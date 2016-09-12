% Checking parameter rescalings

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
I    = 0.0;
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

% Solve model in paper
initState = [0.0;0.0;0.0;0.0];
options = odeset('AbsTol',1e-8,'RelTol',1e-6);
currentTime = 0;
endTime = 1000;
[t,y] = ode45( @IFfunSteve, [currentTime, endTime], initState, options, p);
figure(1);
plotyy(t,y(:,1),t,y(:,2));

% %% Now rescale parameters
% tau = tau/gl;
% gh  = gh*V_h/gl;
% gs  = -gs/gl;
% I   = -I/gl;
% 
% changePars = {'tau','gh','gs','I'};
% for i = 1:length(changePars)
%   p = setfield(p,changePars{i},eval(changePars{i}));
% end
% [T,Y] = ode45( @IFfun, [currentTime, endTime], initState, options, p);
% plot(t,y(:,1),T,Y(:,1),'r.');

%% Now hyerpolarise cells and let loose
tt = t;
yy = y;
currentState = yy(end,:)';
currentTime = endTime;
I = -30.0;
p.I = I;
endTime = currentTime+250.0;
[t,y] = ode45( @IFfunSteve, [currentTime, endTime], currentState, options, p);
tt = [tt;t];
yy = [yy;y];
plot(t,y(:,1));

currentState = yy(end,:)';
currentTime = endTime;
endTime = currentTime+500.0;
I = 0.0;
p.I = I;
[t,y] = ode45( @IFfunSteve, [currentTime, endTime], currentState, options, p);
tt = [tt;t];
yy = [yy;y];

plot(tt,yy(:,1));

%% Now check against my algorithm
N    = 1;
tol  = 1e-6;
epsilon = 1e-4;

% just deal with one neuron for now
globalState = initState;
globalZone  = 2;

% calculate derivatives using fininte differences
fun1 = @(t,v0,n0,u0,y0,thresh) v0*exp(-t/tau)...
  +(beta_left*gh*(tau-tau_h+tau_h*exp(-t/tau_h)))/(tau-tau_h)...
  -(beta_left*gh*tau*exp(-t/tau))/(tau-tau_h)...
  -(gh*n0*tau_h*exp(-t/tau_h))/(tau-tau_h)...
  +(gh*n0*tau_h*exp(-t/tau))/(tau-tau_h)...
  +(gs*exp(-alpha*t).*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha^2*t*tau*y0))/(alpha*tau-1)^2 ...
  -(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(alpha*tau-1)^2 ...
  -I+I*exp(-t/tau)-thresh;
dfun1 = @(t,v0,n0,u0,y0,thresh,f0) (fun1(t+epsilon,v0,n0,u0,y0,thresh)-f0)/epsilon;

fun2  = @(t,v0,n0,u0,y0,thresh) (1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))+tau.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))+tau_h.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))-tau_h.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))).*(-v0+(beta_centre.*tau.^2.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(gamma_centre.*(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*-2.0+gamma_centre.*tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0+gamma_centre.*tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0)+(beta_centre.*tau_h.^2.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(gamma_centre.*(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*-2.0+gamma_centre.*tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0+gamma_centre.*tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0)+(I.*tau_h.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))+(I.*tau_h.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(I.*tau_h.^2.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau.*tau_h.*2.0+tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.^2-tau_h.^2-gamma_centre.*gh.*tau.*tau_h.*4.0)+(beta_centre.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(gamma_centre.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)))-(beta_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(-gamma_centre.*(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+gamma_centre.*tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+gamma_centre.*tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))+(gs.*tau_h.*u0.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)-(beta_centre.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(gamma_centre.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)))-alpha.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0+tau.*tau_h.*exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(t.*tau+t.*tau_h-tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0)+(alpha.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0+tau.*tau_h.*exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(t.*tau+t.*tau_h-tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*t.*tau.*tau_h.*2.0).*2.0).*(1.0./2.0))./tau+alpha.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0-tau.*tau_h.*exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(-t.*tau-t.*tau_h+tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0)+(alpha.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0-tau.*tau_h.*exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(-t.*tau-t.*tau_h+tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+alpha.*t.*tau.*tau_h.*2.0).*2.0).*(1.0./2.0))./tau+(I.*tau.*tau_h.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau.*tau_h.*2.0+tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.^2-tau_h.^2-gamma_centre.*gh.*tau.*tau_h.*4.0)+(I.*tau_h.^2.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(I.*tau.*tau_h.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(gs.*tau_h.^2.*u0.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)-(gs.*tau_h.*u0.*exp(-alpha.*t).*(exp(alpha.*t)-exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)-(beta_centre.*tau.^2.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(gamma_centre.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)))-(beta_centre.*tau_h.^2.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(gamma_centre.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)))+(alpha.*gs.*tau_h.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0+tau.*tau_h.*exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(t.*tau+t.*tau_h-tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./tau-(alpha.*gs.*tau_h.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0-tau.*tau_h.*exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(-t.*tau-t.*tau_h+tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./tau+(gs.*tau.*tau_h.*u0.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)-(gs.*tau_h.^2.*u0.*exp(-alpha.*t).*(exp(alpha.*t)-exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)+(beta_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(gamma_centre.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)))+(gs.*tau.*tau_h.*u0.*exp(-alpha.*t).*(exp(alpha.*t)-exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)).*(-1.0./2.0)+(1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(tau.^2.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))-tau.^2.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))+tau_h.^2.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))-tau_h.^2.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h))-exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.*tau_h.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)-t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*2.0+tau.*tau_h.*exp((t.*tau.*(-1.0./2.0)-t.*tau_h.*(1.0./2.0)+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*(1.0./2.0))./(tau.*tau_h)).*2.0).*(n0+(beta_centre.*tau.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))+(beta_centre.*tau.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(beta_centre.*tau.^2.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau.*tau_h.*2.0+tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.^2-tau_h.^2-gamma_centre.*gh.*tau.*tau_h.*4.0)+(beta_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0))./(tau.*tau_h.*2.0+tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.^2-tau_h.^2-gamma_centre.*gh.*tau.*tau_h.*4.0)+(beta_centre.*tau.^2.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(beta_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))+alpha.*gamma_centre.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0+tau.*tau_h.*exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(t.*tau+t.*tau_h-tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*gamma_centre.*gs.*y0.*(tau.^2.*tau_h.^2.*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*4.0-tau.*tau_h.*exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h)).*1.0./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).^2.*(-t.*tau-t.*tau_h+tau.*tau_h.*2.0+t.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+alpha.*t.*tau.*tau_h.*2.0).*2.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-(I.*gamma_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*2.0)./(tau.*tau_h.*2.0+tau.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)+tau_h.*sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-tau.^2-tau_h.^2-gamma_centre.*gh.*tau.*tau_h.*4.0)+(I.*gamma_centre.*tau.*tau_h.*(exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0)./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0))-(gamma_centre.*gs.*tau.*tau_h.*u0.*(exp((t.*(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0).*(1.0./2.0))./(tau.*tau_h))-1.0).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0)./(tau+tau_h-sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)-(gamma_centre.*gs.*tau.*tau_h.*u0.*exp(-alpha.*t).*(exp(alpha.*t)-exp((t.*(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)).*(1.0./2.0))./(tau.*tau_h))).*1.0./sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0).*2.0)./(tau+tau_h+sqrt(tau.*tau_h.*-2.0+tau.^2+tau_h.^2+gamma_centre.*gh.*tau.*tau_h.*4.0)-alpha.*tau.*tau_h.*2.0)).*(1.0./4.0))./(gamma_centre.*tau)) - thresh;
dfun2 = @(t,v0,n0,u0,y0,thresh,f0) (fun2(t+epsilon,v0,n0,u0,y0,thresh)-f0)/epsilon;

fun3 = @(t,v0,n0,u0,y0,thresh) v0*exp(-t/tau)...
  +(beta_right*gh*(tau-tau_h+tau_h*exp(-t/tau_h)))/(tau-tau_h)...
  -(beta_right*gh*tau*exp(-t/tau))/(tau-tau_h)...
  -(gh*n0*tau_h*exp(-t/tau_h))/(tau-tau_h)...
  +(gh*n0*tau_h*exp(-t/tau))/(tau-tau_h)...
  +(gs*exp(-alpha*t).*(alpha*tau*u0-u0-alpha*t*y0+alpha*tau*y0+alpha^2*t*tau*y0))/(alpha*tau-1)^2 ...
  -(gs*exp(-t/tau)*(alpha*tau*u0-u0+alpha*tau*y0))/(alpha*tau-1)^2 ...
  -I+I*exp(-t/tau)-thresh;
dfun3 = @(t,v0,n0,u0,y0,thresh,f0) (fun3(t+epsilon,v0,n0,u0,y0,thresh)-f0)/epsilon;

% Initialise network
state = initState;
zone = (state(1)<V_left)+2*(V_left<state(1))*(state(1)<V_right)...
    +3*(V_right<state(1));
currentTime = 0.0;
endTime = 1000.0;
eventTime = 0.1;
iterate = 1;

% Collect output
voltageHistory = zeros(endTime/eventTime,1);
gatingHistory  = zeros(endTime/eventTime,1);
zoneHistory    = zeros(endTime/eventTime,1);

while currentTime < endTime

  changeFlag = 0;

  if (zone==1)
    v = fun1(eventTime,state(1),state(2),state(3),state(4),0.0);
    crossTime = eventTime;
    if v > V_left
      changeFlag = 1;
      v  = fun1(crossTime,state(1),state(2),state(3),state(4),V_left);
      dv = dfun1(crossTime,state(1),state(2),state(3),state(4),V_left,v);
      while abs(v)>tol
        crossTime = crossTime - v/dv;
        v  = fun1(crossTime,state(1),state(2),state(3),state(4),V_left);
        dv = dfun1(crossTime,state(1),state(2),state(3),state(4),V_left,v);
      end
    end
    state = UpdateStateZone1(crossTime,state,alpha,beta_left,beta_centre,beta_right,...
                gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    if (changeFlag==1)
      zone = zone+1;
      state(1) = V_left;
      state = UpdateStateZone2(eventTime-crossTime,state,alpha,beta_left,beta_centre,beta_right,...
                gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    end
  elseif (zone==2)
    v = fun2(eventTime,state(1),state(2),state(3),state(4),0.0);
    crossTime = eventTime;
    if v < V_left
      changeFlag = -1;
      v  = fun2(crossTime,state(1),state(2),state(3),state(4),V_left);
      dv = dfun2(crossTime,state(1),state(2),state(3),state(4),V_left,v);
      while abs(v)>tol
        crossTime = crossTime - v/dv;
        v  = fun2(crossTime,state(1),state(2),state(3),state(4),V_left);
        dv = dfun2(crossTime,state(1),state(2),state(3),state(4),V_left,v);
      end
    end

    if v > V_right
      changeFlag = 1;
      v  = fun2(crossTime,state(1),state(2),state(3),state(4),V_right);
      dv = dfun2(crossTime,state(1),state(2),state(3),state(4),V_right,v);
      while abs(v)>tol
        crossTime = crossTime - v/dv;
        v  = fun2(crossTime,state(1),state(2),state(3),state(4),V_right);
        dv = dfun2(crossTime,state(1),state(2),state(3),state(4),V_right,v);
      end
    end
    state = UpdateStateZone2(crossTime,state,alpha,beta_left,beta_centre,beta_right,...
               gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    if (changeFlag==-1)
      zone=zone-1;
      state(1) = V_left;
      state = UpdateStateZone1(eventTime-crossTime,state,alpha,beta_left,beta_centre,...
            beta_right,gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    end
    if (changeFlag==1);
      zone=zone+1;
      state(1) = V_right;
      state = UpdateStateZone3(eventTime-crossTime,state,alpha,beta_left,beta_centre,...
            beta_right,gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    end
  elseif (zone==3)
    v = fun3(eventTime,state(1),state(2),state(3),state(4),0.0);
    crossTime = eventTime;
    if v < V_right
      changeFlag = -1;
      v  = fun3(crossTime,state(1),state(2),state(3),state(4),V_right);
      dv = dfun3(crossTime,state(1),state(2),state(3),state(4),V_right,v);
      while abs(v)>tol
        crossTime = crossTime - v/dv;
        v  = fun3(crossTime,state(1),state(2),state(3),state(4),V_right);
        dv = dfun3(crossTime,state(1),state(2),state(3),state(4),V_right,v);
      end
    end
    if v > V_th
      changeFlag = 1;
      v  = fun3(crossTime,state(1),state(2),state(3),state(4),V_th);
      dv = dfun3(crossTime,state(1),state(2),state(3),state(4),V_th,v);
      while abs(v)>tol
        crossTime = crossTime - v/dv;
        v  = fun3(crossTime,state(1),state(2),state(3),state(4),V_th);
        dv = dfun3(crossTime,state(1),state(2),state(3),state(4),V_th,v);
      end
    end
    state = UpdateStateZone1(crossTime,state,alpha,beta_left,beta_centre,beta_right,...
                gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    if (changeFlag==-1)
      zone = zone-1;
      state(1) = V_right;
      state = UpdateStateZone2(eventTime-crossTime,state,alpha,beta_left,beta_centre,beta_right,...
                gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
    end
    if (changeFlag==1)
      zone = zone+1;
      state(1) = V_r;
      state = UpdateStateZone2(eventTime-crossTime,state,alpha,beta_left,beta_centre,beta_right,...
                gamma_centre,tau,tau_h,gh,gs,I,V_r,N);
      state(1) = V_r;
    end
  end

  currentTime = currentTime + eventTime;
  voltageHistory(iterate) = state(1);
  gatingHistory(iterate)  = state(2);
  zoneHistory(iterate)    = zone;
  iterate = iterate+1;
end

% plot output
time = ((1:length(voltageHistory)))*eventTime;
plot(t,y(:,1),time,voltageHistory,'ro');
