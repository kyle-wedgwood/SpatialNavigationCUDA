close all;
clear all;

%model parameters  
P.C =1; %time constant
P.g_l = 0.25;%0.15;%0.3;
P.alpha =P.g_l/(P.C*5);%0.5;
P.tau_h =800;
P.tau_reset =200;%15;%5;%refractoriness constant
P.V_th =14; %4.5;%11;%threshold
P.V_reset = 0; %reset
P.beta = 0.5;
%h-current parameters  
P.g_h =1*40;%0.25;%0.13;%0.079;%
%synaptic current parameters 
P.g_syn =15;

P.V_half = -10;%
P.k = 10;%
P.V_m = P.V_half-2*P.k;%
P.V_p = P.V_half+2*P.k;%


%connectivity parameters 
P.sigma =25;% 5;%1;%
%spatial parameters
L =10*P.sigma;%
P.N =1000;%;
P.dx =2*L/(P.N-1);%0.075;%
x = -L:P.dx:L;%L:P.dx:-L;%linspace(-L,L,P.N);%
x = x';

X = meshgrid(x);
dif = abs(X - X');
P.w0 = -10;%-10;%-1/(2*P.sigma);
% dif = min(dif,(max(x)-min(x))-dif);
P.W = P.w0*(tanh(P.beta*(P.sigma+dif))+tanh(P.beta*(P.sigma-dif)))/2;

%time 
P.dt = 0.01;%0.005;
t_max =20000;
t_aux = 10000;%length(0:0.0045:t_max);%100000;%
t =zeros(length(0:0.0049:t_max),1);
t(1) = 0;

% initialize variables
V = zeros(P.N,t_aux);
aux_V =P.N:-1:P.N-floor(2*P.sigma/(P.dx));

I_ini = zeros(P.N,1);
% V_plot = zeros(P.N,length(0:0.0049:t_max));
n_ih =zeros(P.N,t_aux);
u = zeros(P.N,t_aux);
y = zeros(P.N,t_aux);

% V_plot(:,1) = V(:,1);

spikes = zeros(10000000,2);
ref_aux = zeros(P.N,1);

index = 1;
index2=0;
aux_index=1;
% V_spike = P.V_th+5; %for plotting 
P.V_ini =-30;%-50;%-100;%
P.t_V_ini =1250;

size_aux = ceil(P.N/2);
tt3 = t(1:index+index2);
VV = V(size_aux,1);
nn = n_ih(size_aux,1);
spikes2 = spikes;
save new4.mat VV nn tt3 x spikes2 -v7.3

while t(index+index2) <= t_max
%     initial hyperpolarisation
    if t(index+index2) >= 1000 && t(index+index2)<P.t_V_ini
      I_ini(aux_V) = P.V_ini;
    end
     if t(index+index2) >= P.t_V_ini
        I_ini = zeros(P.N,1);
    end

    t_new = t(index+index2) + P.dt;
    
    [V_new,n_ih_new,u_new,y_new] = RK2multi(V(:,index),n_ih(:,index),...
        u(:,index),y(:,index),I_ini,P.dt,P);

    aux = find(V_new>= P.V_th);%index of neurons with voltage > threshold
    aux = aux.*(ref_aux(aux)==0);%omite the neurons in refractory period
    aux(aux==0) = [];%remove zeros
    if aux
        t_spike = zeros(P.N,1);
        t_spike(aux) = t(index+index2) + P.dt*((P.V_th-V(aux,index))./(V_new(aux)-V(aux,index)));
        t_new = min(t_spike(aux));
        if t_new <= t(index+index2)
            t_new = t(index+index2) + 1e-10;
        end
        [V_new,n_ih_new,u_new,y_new] = RK2multi(V(:,index),n_ih(:,index),...
            u(:,index),y(:,index),I_ini,t_new-t(index+index2),P);
        aux = find(V_new >= P.V_th);%index of neurons with voltage > threshold
        aux = aux.*(ref_aux(aux)==0);%omite the neurons in refractory period
        aux(aux==0) = [];%remove zeros
%         aux = setdiff(aux,aux_ref);%omite those in the refractory period
        aux_syn = zeros(P.N,1); 
        aux_syn(aux) = 1;%kick y if the neuron is firing
        y_new = y_new + P.alpha*P.dx*P.W*aux_syn;
        spikes(aux_index:aux_index+length(aux)-1,1) = x(aux);
        spikes(aux_index:aux_index+length(aux)-1,2) = t_new*ones(size(aux));
        aux_index=aux_index+length(aux);
    end
    

    aux2 = find(ref_aux>0);
    aux3 = [aux2;aux];

    ref_aux(aux3) =  ref_aux(aux3)  + t_new -t(index+index2);
    V_new(aux3) = P.V_reset;
    aux4 = find(ref_aux>=P.tau_reset);
    ref_aux(aux4) = 0;%release from refractory period
    
%     V_plot(:,index+1+index2) =  V_new;
%     V_plot(aux,index+1+index2) = V_spike;
    
    t(index+1+index2) =  t_new;
    V(:,index+1) =  V_new;
    n_ih(:,index+1) =  n_ih_new;
    u(:,index+1)  =  u_new;
    y(:,index+1) =  y_new;
    index = index+1;
    
    
    if mod(index,t_aux) == 0 
        disp(t(index+index2))
        load('new4.mat')
        VV = [VV V(size_aux,2:end)];
        nn = [nn n_ih(size_aux,2:end)];
        tt3 = t(1:index+index2);
        spikes2 = spikes(1:aux_index,:);
        save new4.mat VV nn tt3 x spikes2 -v7.3
        index2=index2+index-1;
        index = 1;
        V = zeros(P.N,t_aux);
        n_ih =zeros(P.N,t_aux);
        u = zeros(P.N,t_aux);
        y = zeros(P.N,t_aux);
        V(:,1) =  V_new;
        n_ih(:,1) =  n_ih_new;
        u(:,1)  =  u_new;
        y(:,1) =  y_new;
    end
end

% 
% t_max = t(index+index2);
% 
% CM = jet(P.N);
% tt=index+index2;
% figure;
% hold on
% for j =P.N:-100:1
%     plot(V_plot(j,1:tt)-j/5,'Color',CM(j,:))
% end
% axis([0 t_max -P.N/5-50 15])
% set(gca,'FontSize',18)
% xlabel('t')
% ylabel('V')
% 
% 
% 
% figure;
% plot(spikes(1:aux_index,2),spikes(1:aux_index,1),'k.')
% axis([0 t_max min(x) max(x)])
% set(gca,'FontSize',18)
% xlabel('t')
% ylabel('x')
% 
% 
% tt=t_max-2000;tt2=t_max;
% spikes_aux = spikes(spikes(:,2)>=tt & spikes(:,2)<=tt2,:);
% figure;
% hold on
% for j =ceil(P.N/2)-floor(P.sigma/abs(P.dx)):ceil(P.N/2)+floor(P.sigma/abs(P.dx))
%     plot(spikes_aux(spikes_aux(:,1)==x(j),2),spikes_aux(spikes_aux(:,1)==x(j),1),'k.') 
% end
% % axis([tt tt2 x(ceil(P.N/2)-floor(P.sigma/abs(P.dx))) x(ceil(P.N/2)+floor(P.sigma/abs(P.dx)))])
% set(gca,'FontSize',18)
% xlabel('t')
% ylabel('x')
% 
% aux= spikes_aux(spikes_aux(:,1)==x(ceil(P.N/2)),2);
% yy = aux(2:end) - aux(1:end-1);

% 
% j = ceil(P.N/2);    
% tt = find(t>=t_max-200,1);tt=max(1,index-(index+index2-tt));
% tt2 = index;
% figure;
% mma = max(n_ih(j,tt:tt2));
% mmi = min(n_ih(j,tt:tt2));
% plot(V(j,tt:tt2),n_ih(j,tt:tt2),'r')
% hold on 
% line([P.V_th P.V_th],[mmi mma],'Color','b')
% line([P.V_reset P.V_reset],[mmi mma],'Color','g')
% line([P.V_m P.V_m],[mmi mma],'Color','m')
% line([P.V_p P.V_p],[mmi mma],'Color','m')
% set(gca,'FontSize',18)
% xlabel('V')
% ylabel('n')
% title(['n of Neuron at x=', num2str(x(j)), '(t \in [', num2str(t(tt)),',',num2str(t(tt2)), '] ms)']) 
% 
%   
%    
% j=ceil(P.N/2);
% tt = find(t>=t_max-200,1);tt=max(1,index-(index+index2-tt));
% tt2 = index;
% figure;
% [hAx,hLine1,hLine2]=plotyy(t(tt:tt2),V(j,tt:tt2),t(tt:tt2),P.g_h.*n_ih(j,tt:tt2));
% 
% set(hAx,'FontSize',18)
% title(['Zoom of Neuron ', num2str(j)])
% xlabel('time (ms)')
% ylabel(hAx(1),'V')
% ylabel(hAx(2),'I_h')
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tt = find(t>=t_max-1000,1);%tt=max(1,index-(index+index2-tt));
% tt2 = index + index2;
% figure;
% mma = max(nn(tt:tt2));
% mmi = min(nn(tt:tt2));
% plot(VV(tt:tt2),nn(tt:tt2),'r')
% hold on 
% line([P.V_th P.V_th],[mmi mma],'Color','b')
% line([P.V_reset P.V_reset],[mmi mma],'Color','g')
% line([P.V_m P.V_m],[mmi mma],'Color','m')
% line([P.V_p P.V_p],[mmi mma],'Color','m')
% set(gca,'FontSize',18)
% xlabel('V')
% ylabel('n')
% figure;
% [hAx,hLine1,hLine2]=plotyy(t(tt:tt2),VV(tt:tt2),t(tt:tt2),nn(tt:tt2));
% 
% set(hAx,'FontSize',18)
% 
% xlabel('time (ms)')
% ylabel(hAx(1),'V')
% ylabel(hAx(2),'n_h')