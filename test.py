import opentorsion as ot
import numpy as np
import matplotlib.pyplot as plt

shaft1 = ot.Shaft(0, 1, L=None, odl=None, k=25e+6)
shaft2 = ot.Shaft(1, 2, L=None, odl=None, k=25e+6)
shaft3 = ot.Shaft(2, 3, L=None, odl=None, k=25e+6)
shaft4 = ot.Shaft(3, 4, L=None, odl=None, k=25e+6)

'''
To create a disk element, call the ot.Disk() method. As parameters, it takes
the node number where the disk is placed, and the disk's rotational inertia (I).
The method returns the created disk element.
'''

# Below, we create the 4 disk elements in the model
disk1 = ot.Disk(0, I=100) #Compressor
disk2 = ot.Disk(1, I=10)  #Coupling #1
disk3 = ot.Disk(2, I=50)  #Turbine
disk4 = ot.Disk(3, I=10)  #Coupling #2
disk5 = ot.Disk(4, I=80)  #Generator
# To create the assembly, first the created elements are added to lists
shafts = [shaft1, shaft2, shaft3, shaft4]
disks = [disk1, disk2, disk3, disk4, disk5]

# An OpenTorsion model is created with the ot.Assembly() method. The previously defined elements are
# given as parameters. The method returns the assembly object.
compressor_train = ot.Assembly(shafts, disk_elements=disks)
print('toimii')
plot_tool = ot.Plots(compressor_train)
plot_tool.figure_eigenmodes(3)
# The assembly can be visualized using the plot_assembly method