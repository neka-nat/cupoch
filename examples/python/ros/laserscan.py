import numpy as np
import cupoch as cph

scan = cph.geometry.LaserScanBuffer(100)
scan.add_ranges(cph.utility.FloatVector(np.ones(100)))