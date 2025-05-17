#define wait(s) atomic { s > 0 -> s-- }
#define signal(s) s++

int semVecMul = 1;
int semCompute = 1;
int semGradiente = 1;

int wg1 = 0;
int wg2 = 0;
int wg3 = 0;

#define NF 5

proctype vecMul(int i){
  int dot = 0;
  wait(semVecMul);
  printf("calculando el vecmul %d\n", i);
  
  // Simulación del producto punto (simplificado)
  dot = i * 2 + 3;

  printf("vecmul calculado %d, resultado: %d\n", i, dot);
  signal(semVecMul);
  atomic { wg1++ }
}

proctype computeCost(int i){
  int cost = 0;
  wait(semCompute);
  printf("calculando el costo %d\n", i);
  
  // Simulación de la función de costo (simplificada)
  cost = i * i;

  printf("Costo calculado %d, valor: %d\n", i, cost);
  signal(semCompute);
  atomic { wg2++ }
}

proctype calcGradiente(int i){
  int grad = 0;
  wait(semGradiente);
  printf("calculando el gradiente %d\n", i);
  
  // Simulación del cálculo del gradiente (simplificado)
  grad = 2 * i + 1;

  printf("gradiente calculada %d, valor: %d\n", i, grad);
  signal(semGradiente);
  atomic { wg3++ }
}

init {
  int i = 0;

  // Fase 1: vecMul
  atomic {
    i = 0;
    do
    :: i < NF -> run vecMul(i); i++
    :: else -> break
    od
  }
  do :: wg1 == NF -> break od;

  // Fase 2: computeCost
  atomic {
    i = 0;
    do
    :: i < NF -> run computeCost(i); i++
    :: else -> break
    od
  }
  do :: wg2 == NF -> break od;

  // Fase 3: calcGradiente
  atomic {
    i = 0;
    do
    :: i < NF -> run calcGradiente(i); i++
    :: else -> break
    od
  }
  do :: wg3 == NF -> break od;

  printf("época completa\n");
}

