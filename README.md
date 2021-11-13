# keras2cfd-dem
Using pretrained ANN to predict drag forces during runtime in CFD-DEM. The model can be trained on a fine mesh case and used to improve accuracy of coarse mesh modeling.
Base drag force models are:

Di Felice model (Di Felice, R. (1994). The voidage function for fluid-particle interaction systems. International Journal of Multiphase Flow, 20(1), 153-159.
Zhou, Z. Y., Kuang, S. B., Chu, K. W., & Yu, A. B. (2010). Discrete particle simulation of particle–fluid flow: model formulations and their applicability. Journal of Fluid Mechanics, 661, 482-510.)

Koch-Hill model (Hill, R.J., Koch, D.L. and Ladd, A.J.C. (2001) Moderate-Reynolds-Number Flows in Ordered and Random Arrays of Spheres. Journal of Fluid Mechanics, 448, 243-278.)


Implemented within CFDEMproject - https://github.com/CFDEMproject/CFDEMcoupling-PUBLIC
Base code: https://github.com/gosha20777/keras2cpp
