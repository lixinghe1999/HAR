import matplotlib.pyplot as plt

keypoints = {
    'nose': (0.0, 1.6),
    'left-eye': (-0.05, 1.6),
    'right-eye': (0.05, 1.6),
    'left-ear': (-0.1, 1.6),
    'right-ear': (0.1, 1.6),
    'left-shoulder': (-0.2, 1.4),
    'right-shoulder': (0.2, 1.4),
    'left-elbow': (-0.3, 1),
    'right-elbow': (0.3, 1),
    'left-wrist': (-0.25, 0.8),
    'right-wrist': (0.25, 0.8),
    'left-hip': (-0.2, 1),
    'right-hip': (0.2, 1),
    'left-knee': (-0.2, 0.6),
    'right-knee': (0.2, 0.6),
    'left-ankle': (-0.1, 0.2),
    'right-ankle': (0.1, 0.2)
}
error = {
    'nose': 0.1,
    'left-eye': 0.1,
    'right-eye': 0.11,
    'left-ear': 0.12,
    'right-ear': 0.11,
    'left-shoulder': 0.14,
    'right-shoulder': 0.15,
    'left-elbow': 0.29,
    'right-elbow': 0.34,
    'left-wrist': 0.41,
    'right-wrist': 0.47,
    'left-hip':0.29,
    'right-hip': 0.29,
    'left-knee':0.57,
    'right-knee': 0.6,
    'left-ankle': 0.79,
    'right-ankle': 0.8
}
# Create a figure and axis   
fig, ax = plt.subplots(figsize=(8, 8))

# Draw the human model
for joint, (x, y) in keypoints.items():
    ax.scatter(x, y, s=int(error[joint] * 1000), color='red')

connections = [(0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (6, 12), (5, 11), (11, 12),
            (12, 14), (11, 13), (14, 16), (13, 15)]
lines = []
keyjoints = list(keypoints.keys())
for connection in connections:
    start = connection[0]
    end = connection[1]
    ax.plot([keypoints[keyjoints[start]][0], keypoints[keyjoints[end]][0]], [keypoints[keyjoints[start]][1], keypoints[keyjoints[end]][1]], color='black')

ax.set_xlim(-0.5, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Human Model')

# Show the plot
plt.savefig('error_visualization.png')