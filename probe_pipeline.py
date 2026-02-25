import depthai as dai
p = dai.Pipeline()
print('Pipeline create methods:', [n for n in dir(p) if 'create' in n])
node = p.create(dai.node.ColorCamera)
print('ColorCamera node attrs sample:', [n for n in dir(node) if not n.startswith('_')][:80])
print('Has setBoardSocket:', hasattr(node, 'setBoardSocket'))
print('Has preview attr:', hasattr(node, 'preview'))
print('Has requestFullResolutionOutput:', hasattr(node,'requestFullResolutionOutput'))
