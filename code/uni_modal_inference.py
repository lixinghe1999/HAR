from utils.tracking import imu_loading_example, imu_loading, imu_fusion, imu_tracking, plot_sensor, plot_tracking


# imu_file = "../dataset/aiot/long_walk.csv"
# timestamp, gyroscope, accelerometer, delta_time = imu_loading_example(imu_file)

# imu_file = "../dataset/aiot/Lixing_reading_officework-20241029_110111_583.csv"
# imu_file = "../dataset/aiot/imu_test-20241031_131730_676.csv"
# imu_file = "../dataset/aiot/imu_test-20241031_132557_407.csv"
imu_file = "../dataset/aiot/imu_test-20241031_133842_157.csv"

timestamp, gyroscope, accelerometer, delta_time = imu_loading(imu_file)


timestamp, gyroscope, accelerometer, euler, internal_states, acceleration = imu_fusion(timestamp, gyroscope, accelerometer, delta_time, sample_rate=50)

plot_sensor(timestamp, gyroscope, accelerometer, euler, internal_states)

is_moving, velocity, position = imu_tracking(timestamp, acceleration, sample_rate=50)

plot_tracking(timestamp, acceleration, is_moving, velocity, position)