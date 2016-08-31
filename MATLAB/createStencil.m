% function to check stencil for reset conditions

N = 1024;
index = 0:N*N-1;

x = mod(index,N);
y = floor(index/N);

sigma = 5;
domainSize = 120;
dx = 2*domainSize/(N-1);

% convert to spatial coordinates
x = x*dx-domainSize;
y = y*dx-domainSize;

indices = find(sqrt(x.*x+y.*y) < sigma);
middleIndex = N/2*(N+1);
stencil = indices-middleIndex;

% plot results
val = zeros(N,N);
val(middleIndex+stencil) = 1;
imagesc(val);
axis square;

%% construct another way
spatial_extent = floor(sigma/dx);
newStencil = [];
for row = -spatial_extent:spatial_extent
  for col = -spatial_extent:spatial_extent
    if row*row*dx*dx+col*col*dx*dx<sigma*sigma
      newStencil = [ newStencil; row*N+col];
    end
  end
end

% plot results
newVal = zeros(N,N);
newVal(middleIndex+newStencil) = 1;
figure;
imagesc(newVal);
axis square

%% check stencil
% stencil = newStencil;
stencil_x = mod(stencil+spatial_extent,2*spatial_extent)-spatial_extent;
stencil_y = floor(stencil/N);

for local_index = 0:N*N-1
  local_x = mod(local_index,N);
  local_y = floor(local_index/N);
end