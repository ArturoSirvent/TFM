**1 mayo 2023**  

En 1.1Break.../5-eXP... esta la ultima version de las clases de datos y modelo. la carpeta model de 1.1Break tambien tiene las ultimas motificaciones.  
Creo la carpeta 1.2-HyperParameter... donde haré modificaciones sobre copias de model y las clases anterior mencionadas.  


He creado unas clases para la evaluacion etc, y todo eso esta el original en 1.2Hyper... /1- ,asive...ipynb  
Ahora para probar si funciona, lo voy a probar sobre los datos de pump, y los meto en 3-ApplyModel/2.2masive.pynb  Y lo especializao para nuestro caso de pump con 40 canales lo menos.


NOTA: La nomenclatura para los archivos de resultados 1 y 2 es : (lam, sigma_a, sigma_b, clip). Y la diferencia entre ellas es que:  
- results ha sido entrenado por 10 epocas con 4000 muestras y 1e-3 lr
- results2 con 3000 muestras, 210 epocas y 1e-3 lr tbn