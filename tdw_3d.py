from tdw.controller import Controller
from tdw.add_ons.replicant import Replicant

URL = "file:///home/yangqingzheng/HAZARD/src/HAZARD/data/assets/replicant_0"  # symlink to fireman_linux
c = Controller(port=1071)
rep = Replicant( position={"x":0,"y":0,"z":0})
c.add_ons.append(rep)
c.communicate([])
print("Spawn OK")
c.communicate({"$type": "terminate"})
