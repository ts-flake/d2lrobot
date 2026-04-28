import genesis as gs
import numpy as np

gs.init(backend=gs.gpu)

scene = gs.Scene(
	sim_options = gs.options.SimOptions(dt=0.01),
	show_viewer=True
)

plane = scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
	gs.morphs.URDF(
		file='/home/dhan/Desktop/study/d2lrobot/sim/onshape_to_robot/so101_yaw/robot.urdf',
		fixed=True
	)
)

scene.build()

jnt_names = [f'joint{i}' for i in range(1,8)]
dofs_idx = [robot.get_joint(n).dof_idx_local for n in jnt_names]

for i in range(10000):
	robot.control_dofs_position(
		np.array([np.sin(0.01*i), 0, 0, 0, 0, 0, 0,]),
		dofs_idx
	)
	scene.step()
