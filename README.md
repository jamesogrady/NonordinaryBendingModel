#Nonordinary Peridynamic Bending
This repository contains code used to simulate bending in thin features using a nonordinary peridynamic model.
The theoretical development of this model, as well as results produced by this code, are found in the following papers by James O'Grady and John Foster:

  *[Peridynamic beams: A non-ordinary, state-based model](http://dx.doi.org/10.1016/j.ijsolstr.2014.05.014)
  *[Peridynamic plates and flat shells: A non-ordinary, state-based model](http://dx.doi.org/10.1016/j.ijsolstr.2014.09.003)

##PdBending.py

This python script handles the entire simulation. It includes problem initialization, determination of parameters, and both explicit and quasi-static implicit capability. It requires the Trilinos libraries, particularly PyTrilinos, but also Epetra, EpetraExt, Teuchos, NOX, and Isorropia.

##calcforce.c, calcforce.h, calcforce.i

These files are the C implementation of the force evaluation function for an elastic model. because both explicit and implicit simulations require large numbers of force calculations, the force evaluation is coded in C and wrapped by SWIG to produce functions callable in PdBending.py.

##swigit.sh

This bash script wraps the C force evaluations. It will likely need to be modified for your machine.
