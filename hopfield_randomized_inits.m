
%% Hopfield Networks - randomized intial condition
%Welcome Earthlings

% This code sets some random memories in a hopfield network of whatever
% desired node size, then randomly iniatilizes in the state space a bunch
% of times and sees which of the nominal memories (if any) the system lands
% in (as well as outputs a graph of the energy of the system over time in
% steps

% For next session I'll make some tweeks to this code and create a version
% that looks at the same initialization a bunch of times (so we can see
% somewhat the effect of the randomness in which node updates each step)

stored_states = 3; % How many stored states you wnat
length_state = 20; % How long you want the state vectors
threshold = 0; % sets threshold for steps

%% Initialize memories


% Set states for storage
V_stored = randi([0,1],stored_states,length_state);


% This checks that there aren't two states that you stored that are the
% same
if rank(V_stored)<stored_states
    quit
end

% Initialize weight vector
T = zeros(length_state,length_state);


% Create weight vector based on stored states
for i = 1:length_state
    for j = 1:length_state
        for state = 1:stored_states
            
        T(i,j) = T(i,j) + ((2*V_stored(state,i)-1).*(2*V_stored(state,j)-1));
        
        end
    end
    T(i,i) = 0;
end

% looking at all stable states
V_stable = cat(1,V_stored,1-V_stored,zeros(1,length_state),zeros(1,length_state));



%% Where the action happens

% initialize vector that will store running tally of how often a
% particular memory is the final attractor state
results = zeros(2*stored_states+2,1);

% With given memories randomly intialize starting state for n runthroughs
for runthroughs = 1:1000

    %randomize initial condition
    vect = randi([0,1],1,length_state);
    
  % Initialize energy
    Energy_init = 0;
    
    %Calculating initial energy
    for i = 1:length_state
        for j = [1:i-1 i+1:length_state]
           Energy_init = Energy_init + T(i,j)*vect(i)*vect(j);
        end
    end
    
    Energy_init = -(1/2)*Energy_init;

    
    Energy = [Energy_init];
   % this mov through the network and updates on each step
    for steps = 1:150
        SumTemp = 0;
        i = randi(length_state);
        for j = [1:i-1 i+1:length_state]
            SumTemp = SumTemp + T(i,j)*vect(j);
        end
        
        % stores last state as oldtempvecti and creates a new one (actually
            % stepping)
        oldtempvecti = vect(i);
        if SumTemp>threshold 
           vect(i) = 1;
        elseif SumTemp<threshold
           vect(i) = 0;
        end
       
        Delta_energy = SumTemp*(vect(i)-oldtempvecti);
         %keep running tab on energy
        Energy = [Energy (Energy(length(Energy))-Delta_energy)];
            % I know this changes size every step, as of now I understand
            % this may seem silly (there's a fixed number of steps) but in
            % the future this code may be retrofitted stop after a while
            % loop so this should add some future flexbility
    end
    
    
    % check if we ended up in a "nominal memory" (and which memory that
    % was)
    [sucess, ind] = ismember(vect,V_stable,'rows');
    
    % store which nominal memory we landed in if aplicable
    if sucess == 1
        results(ind) = results(ind)+1;
    end
    
    % plot that shit
    subplot(2,1,1)
    plot(Energy)
    title('Energy over steps each iteration')
    shg

    subplot(2,1,2)
    plot(results)
    title('Nominal memory landing histogram (all iterations)')

    

end


Nominal_landing = sum(results);
clear i j oldtempvecti sucess state Delta_energy ind Energy_init SumTemp

