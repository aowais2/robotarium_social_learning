'''
Owais Hamid, 04/06/20
Robot 1 runs in sawtooth formation
Robot 2 runs in Spiral formation
'''

#Import Robotarium Utilities
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from math import sin,cos
import numpy as np

# Experiment Constants
iterations = 6000 #Run the simulation/experiment for 5000 steps (5000*0.033 ~= 2min 45sec)
N=3

reward1 = 0
reward1_loc = []
reward2 = 0
reward2_loc = []

#Robot 1 waypoint definition. Waypoints define a spiral
x_c = []
y_c = []
x=0
y=0
a=.08
b=.08
angle=0

for i in range(100):
	angle = 0.2*i
	x = (a+b * angle) * cos(angle)		#Needs to be between [-1.5,1.5]
	y = (a+b * angle) * sin(angle)		#Needs to be between [-0.9,0.9]
	x_c = np.append(x_c,x)
	y_c = np.append(y_c,y)
	x_c = np.clip(x_c,-1.5,1.5)
	y_c = np.clip(y_c,-0.9,0.9)

waypoints = np.array([x_c,y_c])		#(2,25)

#Waypoint defining sawtooth formation
percent=30.0
TimePeriod=1.0
Cycles=3
dt=0.12
t=np.arange(-Cycles*TimePeriod/2,Cycles*TimePeriod/2,dt); 
pwm= t%TimePeriod<TimePeriod*percent/100 
pwm = pwm.astype(float)
pwm = np.where(pwm==1,0.9,-0.9)

waypoints_1 = np.array([t,pwm])		#(2,25)

close_enough = 0.03; #How close the leader must get to the waypoint to move to the next one.

#Creating rewards with reproducibility
np.random.seed(0)
reward_x = np.random.uniform(-1.5,1.5,50)
np.random.seed(1)	#Create 50 x_coord for reward
reward_y = np.random.uniform(-0.9,0.9,50)
reward_locs = np.array([reward_x,reward_y])	#Reward Locations

#Initialize states
state = 0
state_1 = 0

#Limit maximum linear speed of any robot
magnitude_limit = 0.15

# For computational/memory reasons, initialize the velocity vector
dxi = np.zeros((2,N))

# Initial Conditions to Avoid Barrier Use in the Beginning.
initial_conditions = np.array([[0,0.5,0.3],[0.5, 0.3, 0.1],[0, 0.2, 0.6]])

# Instantiate the Robotarium object with these parameters
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions, sim_in_real_time=True)

# Grab Robotarium tools to do simgle-integrator to unicycle conversions and collision avoidance
# Single-integrator -> unicycle dynamics mapping
_,uni_to_si_states = create_si_to_uni_mapping()
si_to_uni_dyn = create_si_to_uni_dynamics(angular_velocity_limit=np.pi/2)

# Single-integrator barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Single-integrator position controller
agent1_controller = create_si_position_controller(velocity_magnitude_limit=0.15)
agent2_controller = create_si_position_controller(velocity_magnitude_limit=0.15)

# Plot Graph Connections
x = r.get_poses() # Need robot positions to do this.

r.step()

for t in range(iterations):
	# Get the most recent pose information from the Robotarium. The time delay is ~ 0.033s
	x = r.get_poses()
	xi = uni_to_si_states(x)
	for i in range(1,N):
		# Zero velocities
		dxi[:,[i]]=np.zeros((2,1))
	
	waypoint = waypoints[:,state].reshape((2,1))
	ws = waypoints_1[:,state_1].reshape((2,1))

	dxi[:,[0]] = agent1_controller(x[:2,[0]], waypoint)
	dxi[:,[1]] = agent2_controller(x[:2,[1]], ws)
	if np.linalg.norm(x[:2,[0]] - waypoint) < close_enough:
		state = (state + 1)%100	#the denominator needs to be the len(waypoint array)
	if np.linalg.norm(x[:2,[1]] - ws) < close_enough:
		state_1 = (state_1 + 1)%25	#Same with the denominator here

	#Create reward extension scenario
	for j in range(len(reward_locs[0])):
		if np.linalg.norm(x[:2,[0]] - reward_locs[:,j].reshape(2,1)) < close_enough:
			print("Close to",reward_locs[:,j])
			reward1_loc.append(reward_locs[:,j])
			#np.delete(reward_locs,j,1)
		if np.linalg.norm(x[:2,[1]] - reward_locs[:,j].reshape(2,1)) < close_enough:
			print("Close to",reward_locs[:,j])
			reward2_loc.append(reward_locs[:,j])
			#np.delete(reward_locs,j,1)

	#Keep single integrator control vectors under specified magnitude
	# Threshold control inputs
	norms = np.linalg.norm(dxi, 2, 0)
	idxs_to_normalize = (norms > magnitude_limit)
	dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

	#Use barriers and convert single-integrator to unicycle commands
	dxi = si_barrier_cert(dxi, x[:2,:])
	dxu = si_to_uni_dyn(dxi,x)

	# Set the velocities of agents 1,...,N to dxu
	r.set_velocities(np.arange(N), dxu)

	# Iterate the simulation
	r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
reward1_loc = np.unique(reward1_loc,axis=0)
reward2_loc = np.unique(reward2_loc,axis=0)
np.save('reward1.npy',len(reward1_loc))
np.save('reward2.npy',len(reward2_loc))
r.call_at_scripts_end()
