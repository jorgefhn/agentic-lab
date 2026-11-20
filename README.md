Idea de este proyecto (in-process)

En el dataset de ASB hay cinco tecnicas de ataque: 
1.Naive - se pasa la instruccion maliciosa al nodo.
2. Escape Characters. Se introducen caracteres especiales de escape para ver si prospera el ataque.
3. Context Ignorning. El nodo tiene el mandato de ignorar la instruccionm anterior.
4. Fake Completion. Respuesta simulada para hacer creer al agente que la tarea se completó ya.
5. Combined attack. 

Cada uno de estos ataques está traducido a cada uno de los 11 idiomas descritos en las tablas en el dataset de MAPS. Esta es precisamente su aportación inicial, midiendo tasas de exito junto a otros tres benchmarks (no ofensivos) más. 
A nosotros lo que más nos interesa es ASB porque estamos midiendo ataques, por eso empezamos por aqui.

El pipeline experimental es el siguiente. En una topologia de N nodos, y midiendo entre cuatro topologias (chain, tree, star y mesh) y comparando con la ejecucion de single agent, todos los nodos actuan segun el mismo rol, a elegir de entre los 20 roles que hay. Esos roles tienen asignadas una serie de tool benignas, y tools malignas. A cada nodo, se le es pasado en el prompt la posibilidad de elegir entre 2 tools benignas y 1 maligna. El objetivo del ataque es que un nodo pase a otro toda la información, y llegue al último/los últimos nodos (llamados nodos SINK, depende de la topología seran 1 o n, por ejemplo, en Tree son los nodos hoja, en chain es el ultimo nodo, en Star los nodos hoja, y en mesh tenemos que definir cual es el nodo SINK, o si es el ultimo nodo al que se le pase el mensaje tambien). 

Cada agente tiene la posibilidad de discernir si el prompt que esta recibiendo es BENIGN o MALICIOUS. Si es BENIGN y no es el ultimo nodo, lo pasa al siguiente, a no ser que sea el nodo SINK, en cuyo caso, finalizaría el ataque con éxito. Si es MALIGN se corta el flujo (fracaso del ataque). Si el último nodo apercibe MALIGN también es fracaso, aunque los otros N-1 no se hayan dado cuenta de eso. También introduciremos falsos positivos intencionadamente para ver la tasa de error de esto. 

El ASR es la tasa de completitud del ataque para cada una de las tecnicas. El objetivo de este primer estudio con el dataset ASB de MAPS traducido a los 11 idiomas es evaluar la tasa de éxito para cada idioma, con distintos modelos, y sacar conclusiones, analizando su comportamiento y la relación idioma-topología-modelo-numero de nodos y ver como se diluye o amplifica el ataque con respecto a un single-agent attack (que es precisamente lo que ya se midió en MAPS). 

Posteriormente se evaluarán los resultados y podremos pasar a otros datasets para replicar este experimento si fuera necesario.
