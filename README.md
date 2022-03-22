# keras2cfd-dem
Using pretrained ANN to predict drag forces during runtime in CFD-DEM. The model can be trained on a fine mesh case and used to improve accuracy of coarse mesh modeling. 
This code was used in our dJFM paper: https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/highresolution-fluidparticle-interactions-a-machine-learning-approach/6C80409A3FA5427BDF3ADF1B26EFB224

Base drag force models are:

Di Felice model (Di Felice, R. (1994). The voidage function for fluid-particle interaction systems. International Journal of Multiphase Flow, 20(1), 153-159.
Zhou, Z. Y., Kuang, S. B., Chu, K. W., & Yu, A. B. (2010). Discrete particle simulation of particle–fluid flow: model formulations and their applicability. Journal of Fluid Mechanics, 661, 482-510.)

Koch-Hill model (Hill, R.J., Koch, D.L. and Ladd, A.J.C. (2001) Moderate-Reynolds-Number Flows in Ordered and Random Arrays of Spheres. Journal of Fluid Mechanics, 448, 243-278.)

Original calculation loops and equations are intentionally letf in, to easier follow the code. However, they can be removed for faster CPU speed.

Implemented within CFDEMproject - https://github.com/CFDEMproject/CFDEMcoupling-PUBLIC
Base code: https://github.com/gosha20777/keras2cpp
