import quadrotorsim
sim = quadrotorsim.Simulator()
sim.get_config('../config.xml')
sim.reset()
print('==================== Init: ====================')
print(sim.get_sensor())

sim.step([1., 1., 0., 0.], 0.1)
print('==================== After step: ====================')
print(sim.get_sensor())

sim.reset()
print('==================== After reset: ====================')
print(sim.get_sensor())
