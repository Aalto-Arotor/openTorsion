from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as LA
from scipy.sparse import linalg as las

from src.opentorsion.disk_element import Disk
from src.opentorsion.shaft_element import Shaft
from src.opentorsion.gear_element import Gear
from src.opentorsion.assembly import Rotor

assembly = Rotor([Shaft(0,1, 1000, 1)], disk_elements=[Disk(0, 4), Disk(0, 3)])

assembly.campbell_diagram()
